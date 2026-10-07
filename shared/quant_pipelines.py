# какой именно INT8-пайплайн реализован в tinytensor, плюс режим weight-only для сравнения.
#
# три режима, чтобы не путаться
#   weight-only. int8 только хранятся веса, при исполнении они разворачиваются в float,
#   matmul идёт во float32 через BLAS, активации не квантуются вообще.
#   dynamic-range (динамический W8A8). веса int8, активации квантуются на лету на входе каждого
#   Conv/Linear по масштабу из рантайма, matmul целочисленный (int8 в int32), после него сразу
#   деквантизация в float. между слоями данные float.
#   full-integer (статический W8A8). и веса, и активации int8 со статическими масштабами из
#   калибровки, int8 идёт между слоями вместе с ReLU и MaxPool.
#
# в tinytensor (Linear._forward_quant и Conv2d._forward_quant) реализован dynamic-range.
# full-integer сравниваем только через LiteRT (05_export_tflite.py), в numpy его нет
import numpy as np

from tinytensor.core.tensor import Tensor
from tinytensor.nn.conv import Conv2d
from tinytensor.nn.functional import im2col_indices
from tinytensor.nn.linear import Linear
from tinytensor.nn.modules import Module


class WeightOnlyLinear(Linear):
    # int8-веса из уже квантованного Linear, но считаем во float32.
    def __init__(self, src):
        Module.__init__(self)
        self.__dict__.update(src.__dict__)     # w_q, w_scale, b_data, in/out_features ...
        self.quantized = False                 # чтобы Linear.forward не ушёл в int8-ветку

    def forward(self, x):
        w = self.w_q.astype(np.float32) * self.w_scale        # (in,out) * (1,out)
        y = np.asarray(x.data, dtype=np.float32) @ w
        if self.b_data is not None:
            y = y + self.b_data
        return Tensor(y, requires_grad=False, device="cpu")


class WeightOnlyConv2d(Conv2d):
    def __init__(self, src):
        Module.__init__(self)
        self.__dict__.update(src.__dict__)
        self.quantized = False

    def forward(self, x):
        N = x.data.shape[0]
        x_col, out_h, out_w = im2col_indices(
            np.asarray(x.data, dtype=np.float32), self.kh, self.kw,
            padding=self.padding, stride=self.stride)
        w_row = self.w_q.astype(np.float32) * self.w_scale    # (out,K) * (out,1)
        y = w_row @ x_col
        if self.b_data is not None:
            y = y + self.b_data
        y = y.reshape(self.out_channels, out_h, out_w, N).transpose(3, 0, 1, 2)
        return Tensor(y, requires_grad=False, device="cpu")


def to_weight_only(model):
    # заменяет квантованные Conv2d/Linear на weight-only обёртки (in-place по списку слоёв)
    layers = model.net.layers
    for i, l in enumerate(layers):
        if getattr(l, "w_q", None) is None:
            continue
        if isinstance(l, Conv2d):
            layers[i] = WeightOnlyConv2d(l)
        elif isinstance(l, Linear):
            layers[i] = WeightOnlyLinear(l)
    return model


def _gemm_layers(model):
    return [l for l in model.net.layers if isinstance(l, (Conv2d, Linear))]


def describe_pipeline(model, mode):
    # описание выводим из реального состояния модели, а не берём константой.
    # заодно проверяем, что все GEMM-слои квантованы и статических масштабов активаций нет
    gl = _gemm_layers(model)
    has_wq = [getattr(l, "w_q", None) is not None for l in gl]
    static_act = any(hasattr(l, "x_scale") or hasattr(l, "act_scale") for l in gl)
    all_q = all(has_wq)
    none_q = not any(has_wq)

    if none_q:
        return {"pipeline": "float32", "weights": "float32", "activations": "float32",
                "compute": "float32 GEMM (BLAS)", "full_integer": False,
                "batch_dependent": False}
    if not all_q:
        raise RuntimeError("смешанная модель (часть слоёв float, часть int8) не поддерживается")
    if static_act:
        raise RuntimeError("найдены статические масштабы активаций, это full-integer, "
                           "а не тот пайплайн, который описан здесь")

    w_desc = "int8, симметричная поканальная (per-output-channel), диапазон [-127,127], zero_point=0"
    if mode == "weight_only":
        return {
            "pipeline": "int8_weight_only",
            "weights": w_desc,
            "activations": "float32 (не квантуются)",
            "compute": "float32 GEMM после поканального разворачивания весов при каждом вызове",
            "between_layers": "float32",
            "full_integer": False,
            "batch_dependent": False,
            "note": "выигрыш только в размере файла, скорость как у float32 плюс оверхед разворачивания",
        }
    return {
        "pipeline": "int8_dynamic_range_w8a8",
        "weights": w_desc,
        "activations": ("int8, динамическая per-tensor симметричная, scale = max|x|/127 считается "
                        "в рантайме по всему входному батчу каждого Conv/Linear, без калибровки"),
        "compute": "int8 -> int32 matmul (numpy, без BLAS), затем деквантизация в float32",
        "between_layers": "float32 (ReLU, MaxPool и Flatten идут во float, int8 между слоями не течёт)",
        "bias": "float32, добавляется после деквантизации",
        "full_integer": False,
        "batch_dependent": True,
        "note": ("scale зависит от батча, поэтому предсказание для объекта зависит от соседей по батчу "
                 "и accuracy меряется ещё и на batch=1"),
    }


PIPELINE_TAGS = {
    "float32": "float32",
    "int8_dynamic_range_w8a8": "INT8 dynamic-range (W8A8, int32 GEMM)",
    "int8_weight_only": "INT8 weight-only",
}
