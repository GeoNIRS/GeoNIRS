from __future__ import annotations
import csv, hashlib, json, os, sys, zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import h5py, numpy as np

HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[2]


OUT=Path(os.environ.get('GEONIRS_CANONICAL_ROOT', ROOT/'data'/'canonical'))
RAW_ROOT=Path(os.environ.get('GEONIRS_RAW_ROOT', ROOT/'data'/'raw'))
ARCH=Path(os.environ.get('GEONIRS_DS8_ARCHIVE', RAW_ROOT/'DS8'/'FNIRS_SNIRF.zip'))
DS5_CANDIDATES=os.environ.get('GEONIRS_DS5_CANDIDATES')
sys.path.insert(0,str(HERE)); sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/'preprocessing'))



import importlib as _importlib
for _bare,_package in (('adapters','preprocessing.formal_v2.adapters'),
                       ('canonical_core','preprocessing.formal_v2.canonical_core'),
                       ('two_wavelength_snirf_adapter','preprocessing.two_wavelength_snirf_adapter')):
    sys.modules.setdefault(_bare,_importlib.import_module(_package))
from adapters import read_recording, read_ds8_zip_entry
from canonical_core import canonical_preprocess, CHANNEL_MAPPING_SCHEMA, PPF, TARGET_FS
from two_wavelength_snirf_adapter import read_two_wavelength_snirf

def twodict(x):
    return {'dataset':x.dataset,'path':x.path,'intensity':x.intensity,'time':x.time,'sampling_rate':x.sampling_rate,'measurements':x.measurements,'wavelengths':x.wavelengths,'source_positions':x.source_positions,'detector_positions':x.detector_positions,'geometry_dimension':x.geometry_dimension,'length_unit':x.length_unit,'length_unit_raw':x.length_unit_raw,'events':x.events,'metadata':x.metadata,'identity':x.identity,'signal_type':'raw_intensity'}
def jobs():
    out=[]



    if DS5_CANDIDATES is not None:
        with Path(DS5_CANDIDATES).open() as f:
            out += [('DS5',r['RAW_FILE']) for r in csv.DictReader(f) if r['INCLUDE_IN_FORMAL']=='YES']
    out += [('DS7',str(p)) for p in (RAW_ROOT/'DS7'/'source_data').glob('**/*.nirs')]
    with zipfile.ZipFile(ARCH) as z: out += [('DS8',f'{ARCH}::{p}') for p in z.namelist() if p.startswith('FNIRS_SNIRF/') and p.endswith('.snirf')]
    out += [('DS44',str(p)) for p in (RAW_ROOT/'DS44').glob('**/*.snirf')]
    out += [('DS47',str(p)) for p in (RAW_ROOT/'DS47'/'snirf_local-bids').glob('**/*.snirf')]



    for condition in ('finger_tapping', 'resting_state', 'walking'):
        out += [('DS55',str(p)) for p in (RAW_ROOT/'DS55').glob(f'{condition}/**/*.snirf')]
    out += [('DS58',str(p)) for root in (RAW_ROOT/'DS58'/'Pre_Ctrl_Images', RAW_ROOT/'DS58'/'Pre_Interv_Images') for p in Path(root).glob('**/*.snirf')]
    out += [('DS59',str(p)) for p in (RAW_ROOT/'DS59').glob('**/*.snirf')]
    return sorted(out)
def load(ds,src):
    if ds=='DS8': return read_ds8_zip_entry(*src.split('::',1))
    if ds in ('DS44','DS55','DS58'): return twodict(read_two_wavelength_snirf(src,ds))
    return read_recording(src,ds)
def safe(v):
    if isinstance(v,dict): return {str(k):safe(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)): return [safe(x) for x in v]
    if isinstance(v,np.ndarray): return safe(v.tolist())
    if isinstance(v,np.generic): return safe(v.item())
    if isinstance(v,float) and not np.isfinite(v): return None
    return v
def run(job):
    ds,src=job; digest=hashlib.sha256((ds+'|'+src).encode()).hexdigest()[:16]; name=Path(src.split('::')[-1]).stem+'__'+digest+'.h5'; path=OUT/ds/name
    if path.exists():


        try:
            with h5py.File(path, 'r') as old:
                old_task = str(old.attrs.get('task', ''))
                old_id = str(old.attrs.get('recording_id', ''))
            if ds != 'DS55' or old_task in ('finger_tapping', 'resting_state', 'walking') and old_id:
                return {'dataset':ds,'source':src,'status':'EXISTS','output':str(path)}
        except Exception:
            pass
    raw=load(ds,src); res=canonical_preprocess(raw); ident=raw['identity']; path.parent.mkdir(parents=True,exist_ok=True)
    valid_t=np.asarray(res['valid_time_mask'],bool); valid_c=np.asarray(res['valid_channel_mask'],bool); mapping=res['channel_mapping_json']; groups=res['measurement_groups_json']
    provenance={'dataset':ds,'recording_id':ident.get('recording_id',''),'identity':ident,'authoritative_raw_source':src,'raw_source_sha256':raw.get('archive_sha256') or hashlib.sha256(Path(src).read_bytes()).hexdigest() if '::' not in src else raw.get('archive_sha256'),'native_sampling_rate':float(raw['sampling_rate']),'native_wavelengths_nm':raw['wavelengths'].tolist(),'source_detector_mapping':mapping,'geometry_dimension':raw['geometry_dimension'],'geometry_unit':raw['length_unit_raw'],'event_provenance':safe(raw['events']),'canonical_preprocessing_version':'Formal_V2_N_WAVELENGTH_CANONICAL_CORE','target_sampling_rate':TARGET_FS,'normalization':safe(res['normalization']),'include_status':'YES','exclusion_reason':''}
    with h5py.File(path,'w') as f:
        f.create_dataset('hbo',data=res['hbo'],compression='gzip'); f.create_dataset('hbr',data=res['hbr'],compression='gzip'); f.create_dataset('valid_time_mask',data=valid_t.astype('uint8')); f.create_dataset('valid_channel_mask',data=valid_c.astype('uint8')); f.create_dataset('invalid_time_mask',data=(~valid_t).astype('uint8')); f.create_dataset('source_detector_distance',data=res['source_detector_distance']); f.create_dataset('channel_mapping_json',data=np.bytes_(json.dumps(safe(mapping),ensure_ascii=False,allow_nan=False))); f.create_dataset('measurement_groups_json',data=np.bytes_(json.dumps(safe(groups),ensure_ascii=False,allow_nan=False))); f.create_dataset('events_json',data=np.bytes_(json.dumps(safe(res['events']),ensure_ascii=False,allow_nan=False))); f.create_dataset('provenance_json',data=np.bytes_(json.dumps(provenance,ensure_ascii=False,allow_nan=False))); f.create_dataset('qc_json',data=np.bytes_(json.dumps(safe(res['qc']),ensure_ascii=False,allow_nan=False))); suffix='3d' if raw['geometry_dimension']==3 else '2d'; f.create_dataset('source_pos'+suffix,data=raw['source_positions']); f.create_dataset('detector_pos'+suffix,data=raw['detector_positions']); f.attrs.update(dataset=ds,recording_id=ident.get('recording_id',''),subject=ident.get('subject',''),session=ident.get('session',''),run=ident.get('run',''),task=ident.get('task',''),source_signal=src,signal_type='raw_intensity',sampling_rate=TARGET_FS,original_sampling_rate=float(raw['sampling_rate']),canonical_core='preprocessing/formal_v2/canonical_core.py',canonical_ppf=PPF,canonical_target_fs=TARGET_FS,FORMAL_RESULT='YES',PILOT_ONLY='NO',include_status='YES',normalization_count=1)
    return {'dataset':ds,'source':src,'status':'PASS','output':str(path),'raw_samples':int(raw['intensity'].shape[0]),'output_samples':int(res['output_samples']),'channels':int(res['hbo'].shape[1]),'valid_channels':int(valid_c.sum()),'invalid_time':int((~valid_t).sum()),'events':len(res['events'])}
def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=OUT,help='canonical HDF5 tree to write (default: %(default)s)')
    parser.add_argument('--raw-root',type=Path,default=RAW_ROOT,help='root of the raw releases (default: %(default)s)')
    parser.add_argument('--ds8-archive',type=Path,default=None,help='FNIRS_SNIRF.zip holding the DS8 recordings')
    parser.add_argument('--ds5-candidates',type=Path,default=None,help='DS5 candidate table with RAW_FILE / INCLUDE_IN_FORMAL columns')
    parser.add_argument('--workers',type=int,default=8)
    args=parser.parse_args()
    globals().update(OUT=args.output,RAW_ROOT=args.raw_root,DS5_CANDIDATES=args.ds5_candidates,
                     ARCH=args.ds8_archive or args.raw_root/'DS8'/'FNIRS_SNIRF.zip')
    if OUT.exists():
        print('RESUME_EXISTING_OUTPUT_ROOT', OUT, 'existing_h5', len(list(OUT.rglob('*.h5'))), flush=True)
    js=jobs(); print('JOBS',len(js),{d:sum(x[0]==d for x in js) for d in sorted({x[0] for x in js})},flush=True)
    OUT.mkdir(parents=True,exist_ok=True); results=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures={pool.submit(run,j):j for j in js}
        for i,f in enumerate(as_completed(futures),1):
            try: r=f.result()
            except Exception as e: j=futures[f]; r={'dataset':j[0],'source':j[1],'status':'FAIL','error':f'{type(e).__name__}: {e}'}
            results.append(r); print(i,'/',len(js),r['dataset'],r['status'],flush=True)
    (OUT/'formal_v2_full_preprocessing_results.json').write_text(json.dumps(results,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({'total':len(js),'pass':sum(r['status']=='PASS' for r in results),'fail':sum(r['status']=='FAIL' for r in results)},ensure_ascii=False))
if __name__=='__main__': main()
