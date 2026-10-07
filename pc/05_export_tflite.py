# экспорт тех же обученных весов в TFLite/LiteRT, чтобы сравнить с numpy-бэкендом.
# для каждой float-модели из ../shared/models/ делаем три файла
#   <имя>__fp32.tflite       float32, эталон рантайма
#   <имя>__int8_dyn.tflite   dynamic-range, веса int8, активации float (ядра квантуют на лету)
#   <имя>__int8_full.tflite  full-integer, веса и активации int8 со статическими масштабами из
#                            калибровки на 200 объектах обучающей выборки. вход и выход float32,
#                            Quantize и Dequantize сидят внутри графа, так что сценарий честный
#                            (картинка на входе, логиты на выходе)
# неструктурные модели экспортируются только в fp32, int8 поверх них смысла не имеет.
# после экспорта проверяем, что tflite-fp32 совпадает с tinytensor, иначе сравнение бессмысленно.
#
# нужен tensorflow, только на пк (pip install tensorflow-cpu)
#   python 05_export_tflite.py
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import argparse
import glob
import pickle

import numpy as np
import tensorflow as tf

from common import MODELS_DIR, SHARED, TESTSET, CALIB, write_csv, fresh_model
from experiment import numpy_forward_fn, batched_predict, accuracy_report, file_size_kb
from litert_utils import make_litert_fn, runtime_info

ap = argparse.ArgumentParser()
ap.add_argument("--only", nargs="*", help="экспортировать только эти модели, например float32 prune30")
args = ap.parse_args()

OUT_DIR = os.path.join(SHARED, "tflite")
os.makedirs(OUT_DIR, exist_ok=True)

test = np.load(TESTSET)
x_test, y_test = test["x"], test["y"]
x_calib = np.load(CALIB)["x"]  # (200,1,28,28), из train


def keras_from_state(sd):
    # переносим веса из tinytensor в keras. свёртки (out,in,kh,kw) переводим в HWIO (kh,kw,in,out),
    # Linear (in,out) остаётся как есть. tinytensor разворачивает (C,H,W), а keras после NHWC
    # разворачивает (H,W,C), поэтому строки первого Linear переставляем, чтобы не тащить Transpose в граф
    w = lambda i: sd[f"net.layers.{i}.weight"]
    b = lambda i: sd[f"net.layers.{i}.bias"]
    c1, c2 = w(0), w(3)
    fc1 = w(7)
    C = c2.shape[0]
    fc1 = fc1.reshape(C, 5, 5, -1).transpose(1, 2, 0, 3).reshape(C * 25, -1)

    L = tf.keras.layers
    inp = tf.keras.Input(shape=(28, 28, 1))  # батч динамический
    x = L.Conv2D(c1.shape[0], 5, padding="same", activation="relu")(inp)
    x = L.MaxPooling2D(2)(x)
    x = L.Conv2D(c2.shape[0], 5, padding="valid", activation="relu")(x)
    x = L.MaxPooling2D(2)(x)
    x = L.Flatten()(x)
    x = L.Dense(w(7).shape[1], activation="relu")(x)
    x = L.Dense(w(9).shape[1], activation="relu")(x)
    x = L.Dense(w(11).shape[1])(x)
    model = tf.keras.Model(inp, x)
    convs = [l for l in model.layers if isinstance(l, L.Conv2D)]
    dense = [l for l in model.layers if isinstance(l, L.Dense)]
    convs[0].set_weights([c1.transpose(2, 3, 1, 0), b(0).ravel()])
    convs[1].set_weights([c2.transpose(2, 3, 1, 0), b(3).ravel()])
    dense[0].set_weights([fc1, b(7).ravel()])
    dense[1].set_weights([w(9), b(9).ravel()])
    dense[2].set_weights([w(11), b(11).ravel()])
    return model


def calib_gen():
    for i in range(len(x_calib)):
        yield [x_calib[i:i + 1].transpose(0, 2, 3, 1).astype(np.float32)]


def convert(model, precision):
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    if precision != "fp32":
        conv.optimizations = [tf.lite.Optimize.DEFAULT]
    if precision == "int8_full":
        conv.representative_dataset = calib_gen
        conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        # inference_input_type и inference_output_type не трогаем, вход и выход остаются float32
    return conv.convert()


files = sorted(glob.glob(os.path.join(MODELS_DIR, "*.tt")))
report = []
for path in files:
    name = os.path.basename(path)[:-3]
    if args.only and name not in args.only:
        continue
    with open(path, "rb") as f:
        sd = pickle.load(f)
    if not any(k.endswith(".weight") for k in sd):
        print(f"пропускаю {name}, это int8-файл tinytensor (LiteRT квантует сам из float)")
        continue

    tt_model = fresh_model(path)
    tt_preds = batched_predict(numpy_forward_fn(tt_model), x_test, 32)

    kmodel = keras_from_state(sd)
    # проверяем, что логиты keras и tinytensor совпадают, так ловятся ошибки в перестановке весов
    from tinytensor.core.tensor import Tensor
    xs = x_test[:256]
    ref = tt_model(Tensor(xs)).data
    got = kmodel.predict(xs.transpose(0, 2, 3, 1), verbose=0)
    diff = float(np.abs(ref - got).max())
    assert diff < 1e-3, f"{name}, keras не совпал с tinytensor (max|dlogit|={diff:.2e})"

    precs = ["fp32"] if name.startswith("unstruct") else ["fp32", "int8_dyn", "int8_full"]
    for prec in precs:
        blob = convert(kmodel, prec)
        out = os.path.join(OUT_DIR, f"{name}__{prec}.tflite")
        with open(out, "wb") as f:
            f.write(blob)
        fn = make_litert_fn(out, batch=1, threads=1)
        preds = batched_predict(fn, x_test, 1)
        acc = accuracy_report(preds, y_test)["accuracy"]
        agree = float((preds == tt_preds).mean())
        report.append((name, prec, f"{file_size_kb(out):.1f}", f"{acc:.4f}", f"{agree:.4f}",
                       f"{diff:.2e}" if prec == "fp32" else ""))
        print(f"{name:<12} {prec:<9} size={file_size_kb(out):7.1f}КБ  acc={acc * 100:.2f}%  "
              f"совпадение с tinytensor-fp32={agree * 100:.2f}%")

write_csv(os.path.join(SHARED, "tflite_export_report.csv"),
          ["variant", "precision", "size_kb", "accuracy_b1_pc", "agreement_with_tinytensor",
           "max_logit_diff_fp32"], report)
print("\nruntime для проверки на пк", runtime_info())
print(f"готово, файлы в {OUT_DIR}, отчёт в {os.path.join(SHARED, 'tflite_export_report.csv')}")
