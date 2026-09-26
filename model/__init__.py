
from importlib import import_module

__all__ = [
    "Ours",
    "DirectEventCNN1DClassifier",
    "FNIRST",
    "FNIRSTLinearClassifier",
    "FNIRS4D",
    "FNIRS4DLinearClassifier",
]

_EXPORTS = {
    "Ours": ("model.ours", "Ours"),
    "DirectEventCNN1DClassifier": ("model.cnn1d", "DirectEventCNN1DClassifier"),
    "FNIRST": ("model.fnirs_t", "FNIRST"),
    "FNIRSTLinearClassifier": ("model.fnirs_t", "FNIRSTLinearClassifier"),
    "FNIRS4D": ("model.fnirs4d", "FNIRS4D"),
    "FNIRS4DLinearClassifier": ("model.fnirs4d", "FNIRS4DLinearClassifier"),
}


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module_name, attribute = _EXPORTS[name]
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value
