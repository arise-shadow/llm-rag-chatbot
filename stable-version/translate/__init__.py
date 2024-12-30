from .gpu import translate_with_gpu
from .furiosa import translate_with_furiosa
from .one_trans import translate_text

__all__ = [
    "translate_with_gpu",
    "translate_with_furiosa",
    "translate_text",
]