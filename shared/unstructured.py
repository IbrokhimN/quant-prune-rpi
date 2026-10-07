# неструктурный прунинг. зануляем самые маленькие по модулю веса и форму матриц не меняем,
# в отличие от структурного, который режет нейроны. из-за этого на плотном numpy ускорения нет,
# хотя модель разрежена. сжатие появляется только если хранить веса разреженно
import numpy as np
from tinytensor.nn.linear import Linear


def prune_unstructured(model, amount=0.3):
    # зануляет долю amount самых малых по модулю весов в каждом Linear-слое (magnitude pruning).
    # порог общий на все веса слоя, форма матриц не меняется
    for layer in model.net.layers:
        if not isinstance(layer, Linear):
            continue
        w = layer.weight.data
        if amount <= 0:
            continue
        # порог это квантиль |w|, ниже него остаётся ровно amount весов
        thresh = np.quantile(np.abs(w), amount)
        mask = np.abs(w) > thresh  # True там, где вес оставляем
        layer.weight.data = (w * mask).astype(w.dtype)
    return model


def sparsity(model):
    # доля нулевых весов в Linear-слоях (для проверки/логов)
    total = 0
    zeros = 0
    for layer in model.net.layers:
        if not isinstance(layer, Linear):
            continue
        w = layer.weight.data
        total += w.size
        zeros += int((w == 0).sum())
    return zeros / total if total else 0.0
