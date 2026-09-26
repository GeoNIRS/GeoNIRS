from __future__ import annotations

import argparse
import hashlib
import h5py
import json
import math
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[2]
AUTHORITY = ROOT / "benchmark" / "benchmark.yaml"
DOWNSTREAM = Path(__file__).resolve().parent / "downstream.yaml"
SPLITS = ("train", "validation", "test")




if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_downstream_config(path: Path | None = None) -> dict:
    return yaml.safe_load((path or DOWNSTREAM).read_text(encoding="utf-8"))



DOWNSTREAM_CONFIG = load_downstream_config()
PROTOCOL_CFG = DOWNSTREAM_CONFIG["protocol"]
METHODS_CFG = DOWNSTREAM_CONFIG["methods"]
CLASSIFIER_CFG = DOWNSTREAM_CONFIG["classifier"]
GRID_CFG = DOWNSTREAM_CONFIG.get("grid", {})

OURS_METHODS = tuple(DOWNSTREAM_CONFIG["ours_methods"])




AMP_METHODS = ("moment_lora",)
BASELINE_METHODS = tuple(DOWNSTREAM_CONFIG["baseline_methods"])
CLASSICAL_METHODS = tuple(DOWNSTREAM_CONFIG["classical_methods"])
ALL_METHODS = OURS_METHODS + BASELINE_METHODS + CLASSICAL_METHODS


MAE_METHODS = tuple(DOWNSTREAM_CONFIG.get("mae_methods", ()))
SUPPORTED_METHODS = ALL_METHODS + MAE_METHODS




MIN_WINDOW_SAMPLES = {
    str(k): int(v) for k, v in DOWNSTREAM_CONFIG.get("minimum_window_samples", {}).items()
}







SINGLETON_TAIL_MERGE_METHODS = frozenset({"fnirs_4d"})


def load_authority() -> dict:
    return yaml.safe_load(AUTHORITY.read_text(encoding="utf-8"))


def num_classes(dataset_cfg: dict) -> int:
    return len(dataset_cfg["class_counts"])


def tensor_batch(raw: dict, device: torch.device) -> dict:
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in raw.items()}


def iter_batches(loader, indices, batch_size, device, shuffle, seed,
                 merge_singleton_tail=False):
    order = np.array(indices, dtype=np.int64, copy=True)
    if shuffle:
        np.random.default_rng(seed).shuffle(order)
    chunks = [order[start:start + batch_size]
              for start in range(0, len(order), batch_size)]
    chunks = [chunk for chunk in chunks if len(chunk)]
    if merge_singleton_tail and len(chunks) > 1 and len(chunks[-1]) == 1:
        chunks[-2] = np.concatenate([chunks[-2], chunks[-1]])
        chunks.pop()
    for chunk in chunks:
        yield tensor_batch(loader.load_batch(chunk), device)


def split_digest(loader, indices) -> str:
    raw = loader.load_batch(list(indices))
    digest = hashlib.sha256()
    for key in ("hbo", "hbr", "valid"):
        digest.update(np.ascontiguousarray(raw[key]).tobytes())
    return digest.hexdigest()


def index_digest(indices) -> str:
    array = np.ascontiguousarray(np.asarray(indices, dtype=np.int64))
    return hashlib.sha256(array.tobytes()).hexdigest()


class InputRecorder:

    def __init__(self, inner):
        self._inner = inner
        self._current = None
        self._touched: dict[str, np.ndarray] = {}

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def __len__(self):
        return len(self._inner)

    def bind(self, method: str) -> None:
        self._current = method

    def rows(self, method: str) -> np.ndarray:
        mask = self._touched.get(method)
        return np.zeros(0, dtype=np.int64) if mask is None else np.nonzero(mask)[0]

    def load_batch(self, indices, **kw):
        raw = self._inner.load_batch(indices, **kw)
        selected = np.asarray(list(indices), dtype=np.int64)
        mask = self._touched.get(self._current)
        if mask is None:
            mask = np.zeros(len(self._inner), dtype=bool)
            self._touched[self._current] = mask
        mask[selected] = True
        return raw




















GEOMETRY_MODES = ("normal", "zeros", "shuffle")


class GeometryTransform:

    def __init__(self, inner, mode: str, seed: int):
        if mode not in GEOMETRY_MODES:
            raise ValueError(f"unknown geometry mode: {mode}")
        self._inner = inner
        self.mode = mode
        self.seed = int(seed)

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def __len__(self):


        return len(self._inner)

    def load_batch(self, indices, **kw):
        raw = self._inner.load_batch(indices, **kw)


        geometry = raw["geometry"]
        if not torch.is_tensor(geometry):
            raise TypeError("geometry must arrive as a torch tensor")
        if self.mode == "zeros":
            raw["geometry"] = torch.zeros_like(geometry)
        elif self.mode == "shuffle":
            shuffled = torch.empty_like(geometry)
            for row in range(geometry.shape[0]):
                rng = np.random.default_rng([self.seed, int(indices[row])])
                permutation = torch.as_tensor(
                    rng.permutation(geometry.shape[1]), device=geometry.device)
                shuffled[row] = geometry[row][permutation]
            raw["geometry"] = shuffled
        return raw


FAIRNESS_AUDIT = ROOT / "benchmark" / "fair_input" / "CROSS_MODEL_FAIRNESS_AUDIT.json"


def cross_model_audit(dataset: str, fair_input_sha256: str) -> dict:
    record = {
        "script": "scripts/audit_cross_model_fairness.py",
        "report": "benchmark/fair_input/CROSS_MODEL_FAIRNESS_AUDIT.json",
        "present": False,
    }
    if not FAIRNESS_AUDIT.is_file():
        return record
    try:
        audit = json.loads(FAIRNESS_AUDIT.read_text(encoding="utf-8"))
    except Exception as error:
        record["error"] = f"{type(error).__name__}: {error}"
        return record
    entry = (audit.get("datasets") or {}).get(dataset) or {}
    record.update({
        "present": True,
        "generated_from_current_fair_input":
            audit.get("fair_input_sha256") == fair_input_sha256,
        "n_models_identical": entry.get("n_models_identical"),
        "n_models": entry.get("n_models"),
        "errors": entry.get("errors"),
        "static_findings": audit.get("static_findings"),
    })
    return record


def code_fingerprint() -> dict:
    files: dict[str, str] = {}
    for module in list(sys.modules.values()):
        path = getattr(module, "__file__", None)
        if not path:
            continue
        candidate = Path(path)
        if not candidate.is_file():
            continue
        resolved = candidate.resolve()
        try:
            relative = resolved.relative_to(ROOT)
        except ValueError:
            continue
        if "__pycache__" in relative.parts:
            continue
        files[str(relative)] = hashlib.sha256(resolved.read_bytes()).hexdigest()
    digest = hashlib.sha256(
        "\n".join(f"{name}\0{value}" for name, value in sorted(files.items())).encode()
    ).hexdigest()
    return {"code_sha256": digest, "code_files": files}


def class_weights(loader, train_idx, num_classes, device):
    labels = np.asarray(loader.load_batch(list(train_idx))["label"])
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    total = float(counts.sum())
    weights = total / (num_classes * np.maximum(counts, 1.0))
    tensor = torch.tensor(weights, dtype=torch.float32, device=device)
    return tensor, counts.astype(int).tolist()


class WarmupCosine:

    def __init__(self, optimizer, epochs: int, warmup_epochs: int, min_lr: float):
        self.optimizer = optimizer
        self.epochs = int(epochs)
        self.warmup = int(warmup_epochs)
        self.min_lr = float(min_lr)
        self.base = [float(g["lr"]) for g in optimizer.param_groups]
        self.last_epoch = -1
        self.step()

    def _lr(self, base: float, epoch: int) -> float:
        if self.warmup and epoch < self.warmup:
            return base * (epoch + 1) / self.warmup
        span = max(1, self.epochs - self.warmup)
        progress = min(max((epoch - self.warmup) / span, 0.0), 1.0)
        return self.min_lr + 0.5 * (base - self.min_lr) * (1.0 + math.cos(math.pi * progress))

    def step(self) -> None:
        self.last_epoch += 1
        for group, base in zip(self.optimizer.param_groups, self.base):
            group["lr"] = self._lr(base, self.last_epoch)

    def current(self) -> list[float]:
        return [float(g["lr"]) for g in self.optimizer.param_groups]


@torch.no_grad()
def evaluate(model, loader, indices, device, batch_size) -> dict:
    model.eval()
    labels, preds, scores = [], [], []
    for batch in iter_batches(loader, indices, batch_size, device, False, 0):
        logits = model(batch)
        preds.extend(logits.argmax(1).cpu().tolist())
        labels.extend(batch["label"].cpu().tolist())
        if logits.shape[1] == 2:
            scores.extend(F.softmax(logits, dim=1)[:, 1].cpu().tolist())
    if not labels:
        return {"n": 0}

    result = {
        "n": len(labels),
        "balanced_accuracy": float(balanced_accuracy_score(labels, preds)),
        "accuracy": float(accuracy_score(labels, preds)),
        "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
    }

    if scores and len(set(labels)) == 2:
        result["auroc"] = float(roc_auc_score(labels, scores))

    return result



















CHANNEL_DROPOUT_MODES = ("fold_fixed", "per_sample")
CHANNEL_DROPOUT_RATIOS = (0.0, 0.10, 0.25, 0.50)


def channel_dropout_sets(channels, rows, ratio, mode, seed, fold) -> np.ndarray:
    if mode not in CHANNEL_DROPOUT_MODES:
        raise ValueError(f"unknown channel dropout mode: {mode}")
    rng = np.random.default_rng([int(seed), int(fold), int(round(float(ratio) * 1000))])
    count = int(round(float(ratio) * channels))
    dropped = np.zeros((rows, channels), dtype=bool)
    if count <= 0:
        return dropped
    if mode == "fold_fixed":
        dropped[:, rng.choice(channels, size=count, replace=False)] = True
        return dropped
    for row in range(rows):
        dropped[row, rng.choice(channels, size=count, replace=False)] = True
    return dropped


def apply_channel_dropout(batch, dropped) -> None:
    keep = (~dropped)[:, :, None].to(batch["hbo"].dtype)
    batch["hbo"] = batch["hbo"] * keep
    batch["hbr"] = batch["hbr"] * keep
    flag = ~dropped
    for key in ("valid", "valid_time_mask"):
        value = batch.get(key)
        if value is not None and value.dim() == 3:
            batch[key] = value.bool() & flag[:, :, None]
    for key in ("valid_channel_mask", "channel_valid"):
        value = batch.get(key)
        if value is not None:
            batch[key] = value.bool() & flag


@torch.no_grad()
def evaluate_channel_dropout(model, loader, indices, device, batch_size,
                             ratio, mode, seed, fold, channels) -> dict:
    order = np.asarray(indices, dtype=np.int64)
    dropped = channel_dropout_sets(channels, len(order), ratio, mode, seed, fold)
    model.eval()
    labels, preds, scores = [], [], []
    for start in range(0, len(order), batch_size):
        chunk = order[start:start + batch_size]
        batch = tensor_batch(loader.load_batch(chunk), device)
        block = torch.from_numpy(dropped[start:start + len(chunk)]).to(device)
        apply_channel_dropout(batch, block)
        logits = model(batch)
        preds.extend(logits.argmax(1).cpu().tolist())
        labels.extend(batch["label"].cpu().tolist())
        if logits.shape[1] == 2:
            scores.extend(F.softmax(logits, dim=1)[:, 1].cpu().tolist())
    if not labels:
        return {"n": 0}
    out = {
        "n": len(labels),
        "channels_total": int(channels),
        "channels_dropped": int(round(float(ratio) * channels)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, preds)),
        "accuracy": float(accuracy_score(labels, preds)),
        "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
    }


    if scores and len(set(labels)) == 2:
        out["auroc"] = float(roc_auc_score(labels, scores))
    return out


def channel_dropout_ladder(model, loader, indices, device, args, channels) -> dict:
    modes = (CHANNEL_DROPOUT_MODES if args.channel_dropout_modes == "both"
             else (args.channel_dropout_modes,))
    ratios = tuple(args.channel_dropout_ratios)
    ladder = {
        "split": "test",
        "retrained": False,
        "ratios": [float(value) for value in ratios],
        "modes": list(modes),
        "seed": int(args.seed),
        "fold": int(args.fold),
        "rule": ("zero the HbO/HbR columns of the dropped channels and flag them "
                 "invalid; geometry untouched; the same fold always drops the "
                 "same channels"),
        "scores": {},
    }
    for mode in modes:
        for ratio in ratios:
            key = f"{mode}@{ratio:.2f}"
            ladder["scores"][key] = evaluate_channel_dropout(
                model, loader, indices, device, args.batch_size,
                ratio, mode, args.seed, args.fold, channels,
            )
    return ladder


def fit_supervised(model, loader, train_idx, val_idx, device, args,
                   optimizer, scheduler, class_weight, merge_singleton_tail=False):
    params = [p for p in model.parameters() if p.requires_grad]
    best_state, best_score, history = None, -1.0, []
    stale = 0





    amp_enabled = str(getattr(args, "method", "")) in AMP_METHODS
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    for epoch in range(1, args.epochs + 1):
        model.train()
        running, seen = 0.0, 0
        for batch in iter_batches(loader, train_idx, args.batch_size, device, True,
                                  args.seed + epoch,
                                  merge_singleton_tail=merge_singleton_tail):
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp_enabled):
                logits = model(batch)
                loss = F.cross_entropy(
                    logits, batch["label"],
                    weight=class_weight, label_smoothing=args.label_smoothing,
                )
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            running += float(loss) * len(batch["label"])
            seen += len(batch["label"])
        scheduler.step()
        val = evaluate(model, loader, val_idx, device, args.batch_size)
        history.append({"epoch": epoch, "train_loss": running / max(seen, 1),
                        "val_balanced_accuracy": val["balanced_accuracy"],
                        "lr": scheduler.current()})
        if val["balanced_accuracy"] > best_score:
            best_score = val["balanced_accuracy"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                history.append({"epoch": epoch, "early_stop": True,
                                "patience": args.patience})
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return history, best_score


def state_digest(state) -> str:
    value = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        array = tensor.detach().cpu().contiguous().numpy()
        value.update(name.encode())
        value.update(str(tuple(array.shape)).encode())
        value.update(array.tobytes())
    return value.hexdigest()






ROW_FILTERS = {"all": None, "lap": "task-lap", "resting": "task-resting"}


def row_filter_mask(loader, which: str) -> np.ndarray:
    needle = ROW_FILTERS[which]
    with h5py.File(loader.path, "r") as handle:
        sample_ids = handle[loader.dataset]["sample_id"].asstr()[()]
    mask = np.asarray([needle in value for value in sample_ids], dtype=bool)
    if not mask.any():
        raise ValueError(f"row filter {which!r} selected no rows in {loader.dataset}")
    return mask


def row_labels(loader) -> np.ndarray:
    with h5py.File(loader.path, "r") as handle:
        return np.asarray(handle[loader.dataset]["label"][:]).astype(np.int64)


def stratified_train_subset(indices, labels, ratio, seed, fold, dataset):
    ratio = float(ratio)
    if not 0.0 < ratio < 1.0:
        raise ValueError("train ratio must be strictly inside (0, 1)")
    indices = np.asarray(indices, dtype=np.int64)
    labels = np.asarray(labels)
    selection = np.random.default_rng([
        int(seed), int(fold), int(len(dataset)),
        int(hashlib.sha256(dataset.encode("utf-8")).hexdigest()[:8], 16),
    ])
    kept = []
    for label in np.unique(labels[indices]):
        rows = indices[labels[indices] == label]
        shuffled = rows[selection.permutation(len(rows))]
        take = max(1, int(round(ratio * len(rows))))
        kept.append(shuffled[:take])
    return np.sort(np.concatenate(kept))


def build_backbone(data_config, model_config):
    from model.ours import Ours

    embedding_dim = int(dict(model_config).get("embedding_dim", 192))
    if embedding_dim == 192:
        return Ours(data_config, model_config)
    from model.ours_large import OursLarge

    return OursLarge(data_config, model_config)


def build_ours(method, checkpoint_path, num_cls, patch_count, seed=None,
               random_init=False):
    if seed is not None:
        torch.manual_seed(seed)
    from benchmark.downstream.ours_event_classifier import OursEventClassifier, configure_trainability

    cfg = METHODS_CFG[method]
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    backbone = build_backbone(saved["config"]["data"], saved["config"]["model"])
    if not random_init:
        backbone.load_state_dict(saved["model"], strict=True)
    init = {
        "random_init": bool(random_init),
        "initial_backbone_sha256": state_digest(backbone.state_dict()),
        "checkpoint_model_sha256": state_digest(saved["model"]),
        "checkpoint_used_for": ("architecture config only" if random_init
                                else "architecture config and weights"),
    }
    mode = method.split("_", 1)[1]
    model = OursEventClassifier(
        backbone, mode,
        head_variant=str(CLASSIFIER_CFG["head_variant"]),
        dropout=float(CLASSIFIER_CFG["head_dropout"]),
        patch_dropout=float(CLASSIFIER_CFG["patch_dropout"]),
        decoder_prefixes=tuple(cfg.get(
            "frozen_modules", ("hbo_to_hbr_decoder", "hbr_to_hbo_decoder"))),
        num_classes=num_cls, downstream_patch_count=patch_count,
    )
    configure_trainability(model, mode, {"lora": {
        "rank": int(cfg.get("rank", 4)),
        "alpha": float(cfg.get("alpha", 8.0)),
        "targets": str(cfg.get("targets", "all_nn_Linear")),
    }})
    return model, init


def build_mae(method, checkpoint_path, channels, points, num_cls, seed,
              random_init=False):
    from model.jung_inspired_mae import load_mae_checkpoint
    from benchmark.downstream.mae_event_classifier import MAEEventClassifier

    if CLASSIFIER_CFG["head_variant"] != "flatten_linear" or CLASSIFIER_CFG["patch_dropout"] != 0:
        raise ValueError("MAE adapter reviewed for flatten_linear without extra patch dropout")
    torch.manual_seed(seed)
    backbone, saved = load_mae_checkpoint(checkpoint_path, random_init=random_init)
    mode = method.rsplit("_", 1)[1]
    model = MAEEventClassifier(backbone, mode, channels, points, num_cls,
                               dropout=float(CLASSIFIER_CFG["head_dropout"]))
    if mode == "lora":
        from benchmark.downstream.ours_event_classifier import configure_trainability
        cfg = METHODS_CFG[method]
        configure_trainability(model, "lora", {"lora": {
            "rank": int(cfg["rank"]), "alpha": float(cfg["alpha"]),
            "targets": str(cfg["targets"])}})
    return model, {
        "random_init": bool(random_init),
        "model_family": saved["model_family"],
        "initial_backbone_sha256": state_digest(backbone.state_dict()),
        "checkpoint_model_sha256": state_digest(saved["model"]),
        "checkpoint_used_for": ("architecture config only" if random_init
                                else "architecture config and weights"),
        "model_config": saved["model_config"],
        "feature_dimensions": model.classifier_feature_dim,
        "encoder_parameters": sum(p.numel() for n, p in backbone.named_parameters()
                                  if backbone.is_encoder_parameter(n)),
        "head_parameters": sum(p.numel() for p in model.linear.parameters()),
        "lora_linear_modules": (len(getattr(model, "_canonical_lora_linear_names", []))
                                if mode == "lora" else 0),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }


def build_baseline(name, channels, points, num_cls, seed):
    torch.manual_seed(seed)
    if name == "cnn_1d":
        from model.cnn1d import DirectEventCNN1DClassifier
        return DirectEventCNN1DClassifier(
            physical_channels=channels, time_points=points,
            num_classes=num_cls, event_pair_mode=False,
        )
    if name == "fnirs_t":
        from model.fnirs_t import FNIRST, FNIRSTLinearClassifier
        backbone = FNIRST(num_class=num_cls, physical_channels=channels, time_points=points)
        return FNIRSTLinearClassifier(backbone, num_classes=num_cls, event_pair_mode=False)
    if name == "fnirs_4d":
        from model.fnirs4d import FNIRS4D, FNIRS4DLinearClassifier
        backbone = FNIRS4D(num_classes=num_cls, physical_channels=channels, time_points=points)
        return FNIRS4DLinearClassifier(backbone, num_classes=num_cls, event_pair_mode=False)
    if name == "fnirsnet":
        from model.fnirsnet import FNIRSNet, FNIRSNetLinearClassifier
        backbone = FNIRSNet(num_class=num_cls, time_points=points, physical_channels=channels)
        return FNIRSNetLinearClassifier(backbone, num_classes=num_cls, event_pair_mode=False)
    if name == "eegnet":




        from model.eegnet import EEGNetAdaptedClassifier
        return EEGNetAdaptedClassifier(
            physical_channels=channels, time_points=points,
            num_classes=num_cls, event_pair_mode=False,
        )
    if name == "moment_base":





        from model.moment_base import MOMENTBase, MOMENTBaseClassifier
        backbone = MOMENTBase(num_class=num_cls, physical_channels=channels,
                              context_length=512)
        return MOMENTBaseClassifier(backbone)
    if name == "moment_lora":





        from model.moment_base import (
            MOMENTBase, MOMENTBaseClassifier, install_moment_lora,
        )
        lora_cfg = METHODS_CFG.get(name, {})
        backbone = MOMENTBase(num_class=num_cls, physical_channels=channels,
                              context_length=512)
        model = MOMENTBaseClassifier(backbone)
        install_moment_lora(
            model,
            rank=int(lora_cfg.get("rank", 4)),
            alpha=float(lora_cfg.get("alpha", 8.0)),
        )
        return model
    raise ValueError(f"unknown baseline: {name}")


def classical_feature_matrix(name, loader, indices, batch_size, device, dropped=None):
    if name == "mean_slope_lda":
        from model.mean_slope_lda import build_pipeline, paired_features
    else:
        from model.statistical_rbf_svm import build_pipeline, paired_features
    order = np.asarray(indices, dtype=np.int64)
    feats, labels = [], []
    for start in range(0, len(order), batch_size):
        chunk = order[start:start + batch_size]
        batch = tensor_batch(loader.load_batch(chunk), device)
        if dropped is not None:
            block = torch.from_numpy(dropped[start:start + len(chunk)]).to(device)
            apply_channel_dropout(batch, block)
        feats.append(paired_features(
            batch["hbo"].cpu().numpy(), batch["hbr"].cpu().numpy(),
            batch["valid"].cpu().numpy(),
        ))
        labels.append(batch["label"].cpu().numpy())
    return np.concatenate(feats), np.concatenate(labels)


def positive_class_scores(pipeline, x):
    if hasattr(pipeline, "decision_function"):
        raw = np.asarray(pipeline.decision_function(x), dtype=float)
        if raw.ndim == 1:
            return raw
        if raw.ndim == 2 and raw.shape[1] == 2:
            return raw[:, 1]
    if hasattr(pipeline, "predict_proba"):
        proba = np.asarray(pipeline.predict_proba(x), dtype=float)
        if proba.ndim == 2 and proba.shape[1] == 2:
            return proba[:, 1]
    return None


def run_classical(name, loader, train_idx, val_idx, test_idx, device, batch_size):
    if name == "mean_slope_lda":
        from model.mean_slope_lda import build_pipeline, paired_features
    else:
        from model.statistical_rbf_svm import build_pipeline, paired_features

    def matrix(indices):
        return classical_feature_matrix(name, loader, indices, batch_size, device)

    x_train, y_train = matrix(train_idx)
    pipeline = build_pipeline()
    pipeline.fit(x_train, y_train)
    out = {}
    for name_split, idx in (("validation", val_idx), ("test", test_idx)):
        x, y = matrix(idx)
        pred = pipeline.predict(x)
        out[name_split] = {
            "n": int(len(y)),
            "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
            "accuracy": float(accuracy_score(y, pred)),
            "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        }


        if len(set(np.asarray(y).tolist())) == 2:
            scores = positive_class_scores(pipeline, x)
            if scores is not None and len(scores) == len(y):
                out[name_split]["auroc"] = float(roc_auc_score(y, scores))
    return out, pipeline


def classical_channel_dropout_ladder(name, loader, indices, pipeline, device,
                                     batch_size, args, channels) -> dict:
    modes = (CHANNEL_DROPOUT_MODES if args.channel_dropout_modes == "both"
             else (args.channel_dropout_modes,))
    ratios = tuple(args.channel_dropout_ratios)
    ladder = {"split": "test", "retrained": False,
              "ratios": [float(value) for value in ratios], "modes": list(modes),
              "seed": int(args.seed), "fold": int(args.fold),
              "rule": ("drop channels by zeroing hbo/hbr and flagging them "
                       "invalid, then recompute the paired hand features; the "
                       "closed-form model itself is never refitted"),
              "scores": {}}
    for mode in modes:
        for ratio in ratios:
            dropped = channel_dropout_sets(channels, len(indices), ratio, mode,
                                           args.seed, args.fold)
            x, y = classical_feature_matrix(name, loader, indices, batch_size,
                                            device, dropped)
            pred = pipeline.predict(x)
            entry = {
                "n": int(len(y)),
                "channels_total": int(channels),
                "channels_dropped": int(round(float(ratio) * channels)),
                "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
                "accuracy": float(accuracy_score(y, pred)),
                "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
            }
            if len(set(np.asarray(y).tolist())) == 2:
                scores = positive_class_scores(pipeline, x)
                if scores is not None and len(scores) == len(y):
                    entry["auroc"] = float(roc_auc_score(y, scores))
            ladder["scores"][f"{mode}@{ratio:.2f}"] = entry
    return ladder


def run(args):



    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    config = load_authority()
    dataset_cfg = config["datasets"][args.dataset]
    n_cls = num_classes(dataset_cfg)
    from benchmark.downstream.fair_input import FormalFairInputLoader

    loader = FormalFairInputLoader(args.dataset, ROOT)
    channels, points = loader.tensor_shape



    if args.geometry_mode != "normal":
        loader = GeometryTransform(loader, args.geometry_mode, args.seed)
    if args.method in MAE_METHODS:
        if not getattr(args, "enable_mae_training", False):
            raise ValueError("the Jung-inspired MAE arms are opt-in: pass --enable-mae-training")
        if not getattr(args, "mae_checkpoint", None):
            raise ValueError("the MAE arms need their own --mae-checkpoint; there is no GeoNIRS fallback")
        if args.checkpoint:
            raise ValueError("use --mae-checkpoint for a MAE arm, not --checkpoint")
    checkpoint_path = (Path(args.mae_checkpoint).resolve() if args.method in MAE_METHODS
                       else Path(args.checkpoint).resolve() if args.checkpoint
                       else ROOT / config["model"]["checkpoints"]["base"]["path"])
    if args.method in MAE_METHODS:
        checkpoint_role = "explicit vanilla-MAE checkpoint (not the GeoNIRS checkpoint)"
    elif not args.checkpoint:
        checkpoint_role = "released GeoNIRS-base checkpoint (authority default)"
    else:
        stem = checkpoint_path.stem
        if stem.startswith("epoch_") and stem[6:].isdigit():
            index = int(stem[6:])
            checkpoint_role = (f"pre-training checkpoint epoch_%03d "
                               f"({index + 1} training passes)" % index)
        elif "ssl" in checkpoint_path.parts or "target_ssl" in str(checkpoint_path):
            checkpoint_role = "target-domain SSL result"
        else:
            checkpoint_role = "explicitly supplied checkpoint"
    checkpoint_identity = {
        "path": str(checkpoint_path),
        "sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        "role": checkpoint_role,
    }




    checkpoint_is_needed = args.method == "all" or args.method in OURS_METHODS + MAE_METHODS
    if checkpoint_is_needed:
        saved_meta = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        patch_samples = int(saved_meta["model_config"]["patch_samples"]
                            if args.method in MAE_METHODS
                            else saved_meta["config"]["data"]["patch_samples"])
        patch_count = points // patch_samples
        if args.method in MAE_METHODS and points % patch_samples:
            raise ValueError("the MAE window must be patch-divisible without touching the Fair Input")
    else:
        patch_count = None
    device = torch.device(args.device)
    indices = {s: loader.split_indices(args.fold, s) for s in SPLITS}




    if args.row_filter != "all":
        keep = row_filter_mask(loader, args.row_filter)
        indices = {s: np.asarray([i for i in v if keep[i]], dtype=np.int64)
                   for s, v in indices.items()}
        empty = [s for s, v in indices.items() if len(v) == 0]
        if empty:
            raise ValueError(f"row filter {args.row_filter!r} emptied {empty}")



    train_ratio_record = None
    if args.train_ratio < 1.0:
        labels = row_labels(loader)
        before = int(len(indices["train"]))
        indices["train"] = stratified_train_subset(
            indices["train"], labels, args.train_ratio,
            args.seed, args.fold, args.dataset,
        )
        kept_labels = labels[indices["train"]]
        class_counts = {str(int(label)): int((kept_labels == label).sum())
                        for label in np.unique(kept_labels)}
        train_ratio_record = {
            "ratio": float(args.train_ratio),
            "train_rows_before": before,
            "train_rows_after": int(len(indices["train"])),
            "train_rows_per_class": class_counts,
            "rule": ("class-balanced prefix of a single seeded permutation of each "
                     "class, applied to the TRAIN split only; validation and test "
                     "keep every row"),
        }


    loader = InputRecorder(loader)







    protocol = {
        "source": str(DOWNSTREAM.relative_to(ROOT)),
        "epochs": args.epochs,
        "early_stop_patience": args.patience,
        "batch_size": args.batch_size,
        "optimizer": str(PROTOCOL_CFG["optimizer"]),
        "warmup_epochs": args.warmup_epochs,
        "lr_schedule": str(PROTOCOL_CFG["scheduler"]),
        "min_lr": args.min_lr,
        "weight_decay": args.weight_decay,
        "grad_clip": args.grad_clip,
        "loss": ("class_weighted_cross_entropy" if args.class_weighted_loss
                 else "cross_entropy"),
        "class_weight_rule": str(PROTOCOL_CFG["class_weight_rule"]),
        "label_smoothing": args.label_smoothing,
        "seed": args.seed,
        "selection_metric": str(PROTOCOL_CFG["selection_metric"]),
        "classifier_head": dict(CLASSIFIER_CFG),
        "input_policy": str(PROTOCOL_CFG["input_policy"]),
        "fair_input_sha256": loader.tensor_sha256,
        "fair_input_policy": loader.policy_name,
        "fair_input_records_sha256": loader.records_sha256,
        "checkpoint": checkpoint_identity,
        "geometry_mode": str(args.geometry_mode),
        "train_ratio": float(args.train_ratio),
        "channel_dropout": (
            {"enabled": False} if not args.channel_dropout else
            {"enabled": True,
             "ratios": [float(value) for value in args.channel_dropout_ratios],
             "modes": (list(CHANNEL_DROPOUT_MODES)
                       if args.channel_dropout_modes == "both"
                       else [args.channel_dropout_modes]),
             "applies_to": list(OURS_METHODS + BASELINE_METHODS + MAE_METHODS),
             "retrained": False}
        ),
    }
    split_sizes = {s: int(len(v)) for s, v in indices.items()}

    methods = ALL_METHODS if args.method == "all" else (args.method,)
    if args.exclude_methods:
        excluded = set(args.exclude_methods)
        methods = tuple(method for method in methods if method not in excluded)
    result = {
        "dataset": args.dataset,
        "fold": args.fold,
        "num_classes": n_cls,
        "tensor_shape": list(loader.tensor_shape),
        "patch_count": patch_count,
        "split_sizes": split_sizes,
        "split_digest": {s: split_digest(loader, v) for s, v in indices.items()},
        "split_index_digest": {s: index_digest(v) for s, v in indices.items()},
        "seed": args.seed,
        "protocol": dict(protocol),
        "methods": {},
    }
    if args.row_filter != "all":
        result["row_filter"] = {
            "name": args.row_filter,
            "rule": f"sample_id contains {ROW_FILTERS[args.row_filter]!r}",
            "rows_kept": int(sum(len(v) for v in indices.values())),
            "rows_in_dataset": int(len(loader)),
        }
    if train_ratio_record is not None:
        result["train_ratio"] = train_ratio_record

    for method in methods:
        started = time.time()
        loader.bind(method)


        entry: dict = {"method": method, "protocol": dict(protocol),
                       "split_sizes": dict(split_sizes)}
        minimum = MIN_WINDOW_SAMPLES.get(method)
        if minimum is not None and points < minimum:
            entry["status"] = "INCOMPATIBLE_WINDOW"
            entry["reason"] = (
                f"{method} requires at least {minimum} time samples; "
                f"{args.dataset} Fair Input window is {points}"
            )
            entry["seconds"] = 0.0
            result["methods"][method] = entry
            print(f"[{args.dataset} f{args.fold}] {method:22} INCOMPATIBLE_WINDOW "
                  f"(window {points} < {minimum})")
            continue
        try:
            if method in CLASSICAL_METHODS:
                metrics, classical_pipeline = run_classical(
                    method, loader, indices["train"], indices["validation"],
                    indices["test"], device, args.batch_size)
                entry.update(metrics)
                if args.channel_dropout:
                    entry["channel_dropout"] = classical_channel_dropout_ladder(
                        method, loader, indices["test"], classical_pipeline,
                        device, args.batch_size, args, channels)
            else:
                if method in MAE_METHODS:
                    model, init = build_mae(method, checkpoint_path, channels, points,
                                            n_cls, args.seed,
                                            random_init=args.random_init)
                    entry["mae_initialization"] = init
                    model.to(device)
                    from benchmark.downstream.ours_event_classifier import (
                        build_optimizer, fit_standardizer)
                    train_batches = list(iter_batches(loader, indices["train"], args.batch_size,
                                                      device, False, args.seed))
                    std = fit_standardizer(model, train_batches, device)
                    entry["standardizer"] = {"near_constant_dimensions": std["near_constant_dimensions"]}
                    cfg = METHODS_CFG[method]
                    optimizer, audit = build_optimizer(
                        model, method.rsplit("_", 1)[1],
                        head_lr=float(cfg["head_lr"]),
                        backbone_lr=float(cfg.get("backbone_lr", cfg["head_lr"])),
                        lora_lr=float(cfg.get("lora_lr", cfg["head_lr"])),
                        weight_decay=float(args.weight_decay),
                    )
                    entry["optimizer_groups"] = audit
                elif method in OURS_METHODS:
                    model, init = build_ours(method, checkpoint_path, n_cls,
                                             patch_count, seed=args.seed,
                                             random_init=args.random_init)
                    if args.random_init:
                        entry["random_init"] = init


                    model.to(device)
                    from benchmark.downstream.ours_event_classifier import (
                        build_optimizer, fit_standardizer)
                    train_batches = list(iter_batches(loader, indices["train"], args.batch_size,
                                                      device, False, args.seed))
                    std = fit_standardizer(model, train_batches, device)
                    entry["standardizer"] = {"near_constant_dimensions": std["near_constant_dimensions"]}


                    cfg = METHODS_CFG[method]
                    optimizer, audit = build_optimizer(
                        model, method.rsplit("_", 1)[1],
                        head_lr=float(cfg["head_lr"]),
                        backbone_lr=float(cfg.get("backbone_lr", cfg["head_lr"])),
                        lora_lr=float(cfg.get("lora_lr", cfg["head_lr"])),
                        weight_decay=float(args.weight_decay),
                    )
                    entry["optimizer_groups"] = audit
                else:
                    model = build_baseline(method, channels, points, n_cls, args.seed)
                    model.to(device)
                    params = [p for p in model.parameters() if p.requires_grad]
                    baseline_lr = float(METHODS_CFG[method]["lr"])
                    optimizer = torch.optim.AdamW(
                        params, lr=baseline_lr, weight_decay=args.weight_decay
                    )
                    entry["optimizer_groups"] = [{
                        "group_name": "baseline_all",
                        "lr": baseline_lr,
                        "weight_decay": args.weight_decay,
                        "parameters": sum(p.numel() for p in params),
                    }]
                scheduler = WarmupCosine(
                    optimizer, args.epochs, args.warmup_epochs, args.min_lr
                )
                if args.class_weighted_loss:
                    weight, counts = class_weights(loader, indices["train"], n_cls, device)
                    entry["class_counts_train"] = counts
                    entry["class_weight"] = [round(float(value), 6) for value in weight]
                else:
                    weight = None
                history, best = fit_supervised(model, loader, indices["train"],
                                               indices["validation"], device, args,
                                               optimizer, scheduler, weight,
                                               merge_singleton_tail=(method in SINGLETON_TAIL_MERGE_METHODS))
                entry["history"] = history
                entry["best_val_balanced_accuracy"] = float(best)
                entry["validation"] = evaluate(model, loader, indices["validation"],
                                               device, args.batch_size)
                entry["test"] = evaluate(model, loader, indices["test"], device, args.batch_size)





                if args.channel_dropout:
                    entry["channel_dropout"] = channel_dropout_ladder(
                        model, loader, indices["test"], device, args, channels
                    )
        except Exception as error:
            entry["error"] = f"{type(error).__name__}: {error}"
        entry["seconds"] = round(time.time() - started, 2)
        result["methods"][method] = entry
        if "error" in entry:
            status = "ERROR " + entry["error"]
        elif "test" in entry:
            status = f"test_bacc={entry['test']['balanced_accuracy']:.4f}"
        else:
            status = entry.get("status", "NO_METRICS")
        print(f"[{args.dataset} f{args.fold}] {method:22} {status}")










    echoes = {name: json.dumps(entry["protocol"], sort_keys=True)
              for name, entry in result["methods"].items()}
    identical_protocol = len(set(echoes.values())) == 1
    distinct_split_digests = len(set(result["split_index_digest"].values())) == 3

    graded = [name for name, entry in result["methods"].items()
              if name in OURS_METHODS + BASELINE_METHODS + MAE_METHODS and "error" not in entry
              and entry.get("status") != "INCOMPATIBLE_WINDOW"]
    evaluated = [name for name, entry in result["methods"].items()
                 if "error" not in entry
                 and entry.get("status") != "INCOMPATIBLE_WINDOW"]

    declared_rows = np.sort(np.concatenate([indices[s] for s in SPLITS]))
    declared_digest = index_digest(declared_rows)
    observed = {name: index_digest(loader.rows(name)) for name in evaluated}



    identical_split_indices = (not observed) or all(
        digest == declared_digest for digest in observed.values()
    )

    split_sets = {s: set(indices[s].tolist()) for s in SPLITS}
    overlaps = {f"{a}__{b}": len(split_sets[a] & split_sets[b])
                for a, b in (("train", "validation"), ("train", "test"),
                             ("validation", "test"))}
    splits_are_disjoint = all(count == 0 for count in overlaps.values())

    result["fairness"] = {
        "identical_input_rows": identical_split_indices,
        "identical_split_indices": identical_split_indices,
        "splits_are_disjoint": splits_are_disjoint,
        "identical_protocol_across_all_methods": identical_protocol,
        "gradient_trained_methods": graded,
        "same_budget_for_all_gradient_methods": all(
            entry["protocol"] == protocol for entry in result["methods"].values()
        ),

        "evaluated_methods": evaluated,
        "declared_row_count": int(len(declared_rows)),
        "declared_row_index_digest": declared_digest,
        "declared_row_bytes_digest": split_digest(loader, declared_rows),
        "per_method_row_index_digest": observed,
        "split_overlap_counts": overlaps,
        "distinct_split_digests": distinct_split_digests,
        "byte_level_cross_model_audit":
            cross_model_audit(args.dataset, protocol["fair_input_sha256"]),
    }
    assert identical_protocol, "methods did not receive an identical protocol"
    assert splits_are_disjoint, "train/validation/test row indices are not disjoint"
    assert identical_split_indices, "methods did not read identical input rows"




    result["runtime"] = {
        "device": args.device,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda": torch.version.cuda,
        "torch_cuda_available": bool(torch.cuda.is_available()),
        "seed_applied_to": ["torch", "numpy", "build_ours", "build_baseline", "build_mae"],
        **code_fingerprint(),
    }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=False)
    parser.add_argument("--fold", type=int, choices=range(1, 6), default=1)
    parser.add_argument("--method", default="all")
    parser.add_argument("--exclude-methods", nargs="*", default=None,
                        help="drop these methods from the roster this run would "
                             "otherwise use. Used to re-run a dataset without an "
                             "expensive method; the result file simply has no "
                             "entry for it instead of an error one.")


    parser.add_argument("--epochs", type=int, default=PROTOCOL_CFG["epochs"])
    parser.add_argument("--patience", type=int, default=PROTOCOL_CFG["early_stopping_patience"],
                        help="early-stop after this many epochs without validation improvement")
    parser.add_argument("--batch-size", type=int, default=PROTOCOL_CFG["batch_size"])
    parser.add_argument("--warmup-epochs", type=int, default=PROTOCOL_CFG["warmup_epochs"])
    parser.add_argument("--min-lr", type=float, default=PROTOCOL_CFG["min_lr"])
    parser.add_argument("--label-smoothing", type=float, default=PROTOCOL_CFG["label_smoothing"])
    parser.add_argument("--no-class-weighted-loss", dest="class_weighted_loss",
                        action="store_false",
                        default=PROTOCOL_CFG["loss"] == "class_weighted_cross_entropy")
    parser.add_argument("--weight-decay", type=float, default=PROTOCOL_CFG["weight_decay"])
    parser.add_argument("--grad-clip", type=float, default=PROTOCOL_CFG["gradient_clip"])
    parser.add_argument("--seed", type=int, default=PROTOCOL_CFG["seed"])
    parser.add_argument("--random-init", action="store_true",
                        help="ours_* only: keep the freshly drawn backbone "
                             "instead of loading the checkpoint weights. The "
                             "architecture config is still read from the "
                             "checkpoint, so this changes the initialisation and "
                             "nothing else. Off by default; with the flag off "
                             "this program behaves exactly as before.")
    parser.add_argument("--row-filter", default="all",
                        choices=sorted(ROW_FILTERS),
                        help="train on a subset of the dataset's rows, selected "
                             "from the frozen Fair Input by recording condition "
                             "(e.g. 'lap' keeps only the task recordings of a "
                             "dataset that also contains resting state). 'all' "
                             "is the default and changes nothing.")
    parser.add_argument("--train-ratio", type=float, default=1.0,
                        help="label-efficiency control: train on a class-balanced "
                             "subset of the TRAIN split only, keeping the same "
                             "training subjects. 1.0 is the default and changes "
                             "nothing; 0.10/0.25/0.50 give the label-efficiency "
                             "curve. Validation and test always keep every row.")
    parser.add_argument("--channel-dropout", action="store_true",
                        help="ours_* only: after normal training, re-score the "
                             "frozen test split with a fraction of channels "
                             "zeroed and marked invalid. Evaluation only - the "
                             "trained model and the reported test score are "
                             "unchanged. Off by default.")
    parser.add_argument("--channel-dropout-ratios", type=float, nargs="+",
                        default=list(CHANNEL_DROPOUT_RATIOS),
                        help=f"drop fractions to evaluate (default "
                             f"{list(CHANNEL_DROPOUT_RATIOS)})")
    parser.add_argument("--channel-dropout-modes", default="both",
                        choices=("both",) + CHANNEL_DROPOUT_MODES,
                        help="fold_fixed: one channel set per (dataset, fold, "
                             "ratio); per_sample: an independent set per test "
                             "row; both runs both ladders")
    parser.add_argument("--checkpoint", default=None,
                        help="ours_* only: fine-tune from this checkpoint instead "
                             "of the authority's formal checkpoint. Used by the "
                             "target-domain SSL control; the file is hashed into "
                             "the result. Default (unset) is the formal "
                             "checkpoint and changes nothing.")
    parser.add_argument("--geometry-mode", default="normal",
                        choices=GEOMETRY_MODES,
                        help="channel-geometry ablation. 'normal' (default) passes "
                             "the frozen geometry through untouched. 'zeros' "
                             "replaces every channel's coordinate vector with "
                             "zeros; 'shuffle' permutes the coordinate vectors "
                             "across channels, keeping capacity and information "
                             "content but destroying the physics. Only ours_* "
                             "reads geometry, so this is a no-op for the other "
                             "seven methods.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--mae-checkpoint", default=None,
                        help="checkpoint written by pretrain/train_mae.py; the MAE "
                             "arms never fall back to the GeoNIRS checkpoint")
    parser.add_argument("--enable-mae-training", action="store_true",
                        help="explicitly enable the Jung-inspired MAE arms; they "
                             "are never implied by --method all")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--output-name", default=None,
                        help="override the result filename inside --output-dir "
                             "(default: <dataset>_fold<fold>.json); used when "
                             "several methods share one output directory")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    config = load_authority()
    datasets = list(config["paper_benchmark"]["datasets"])
    if args.list:
        print("datasets:", " ".join(datasets))
        print("methods :", " ".join(ALL_METHODS))
        if MAE_METHODS:
            print("MAE opt-in (not in all):", " ".join(MAE_METHODS))
        for name in datasets:
            cfg = config["datasets"][name]
            print(f"  {name:5} classes={num_classes(cfg)} task={cfg['task']}")
        return
    if args.dataset is None:
        parser.error("--dataset is required unless --list is used")
    if args.method != "all" and args.method not in SUPPORTED_METHODS:
        parser.error(f"--method must be one of {SUPPORTED_METHODS} or 'all'")

    result = run(args)
    payload = json.dumps(result, indent=2)
    print(payload)
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        name = args.output_name or f"{args.dataset}_fold{args.fold}.json"
        out = args.output_dir / name
        out.write_text(payload + "\n", encoding="utf-8")
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
