# делаем все версии модели из эталона и меряем точность на пк. запускать после 01_train_baseline.py.
# на выходе ../shared/models/*.tt и ../shared/accuracy.csv
#   float32      эталон как есть
#   float32_ft3  эталон и такое же дообучение (+3 эпохи), как у прунинговых версий, чтобы бюджет был честный
#   int8         посттренировочная квантизация эталона (dynamic-range W8A8, см. shared/quant_pipelines.py)
#   prune10..70  структурный прунинг (режет нейроны Linear-слоёв) и дообучение
#   python 02_make_variants.py [--epochs 3]
import argparse
import os
import shutil

from common import (BASELINE, MODELS_DIR, SHARED, USE_CUDA, fresh_model, finetune, load_data,
                    write_csv, FINETUNE_EPOCHS)
from experiment import (numpy_forward_fn, batched_predict, accuracy_report, layer_shapes,
                        file_size_kb)

ap = argparse.ArgumentParser()
ap.add_argument("--epochs", type=int, default=FINETUNE_EPOCHS, help="эпох дообучения (по умолчанию 3)")
args = ap.parse_args()
print(f"дообучение на {'cuda' if USE_CUDA else 'cpu'}, эпох {args.epochs}")

(x_train, y_train), (x_test, y_test) = load_data()
os.makedirs(MODELS_DIR, exist_ok=True)
rows = []


def evaluate_and_save(name, m, precision):
    fn = numpy_forward_fn(m)
    a1 = accuracy_report(batched_predict(fn, x_test, 1), y_test)["accuracy"]
    a32 = accuracy_report(batched_predict(fn, x_test, 32), y_test)["accuracy"]
    path = os.path.join(MODELS_DIR, f"{name}.tt")
    m.save(path)
    rows.append((name, precision, f"{a1:.4f}", f"{a32:.4f}", f"{file_size_kb(path):.1f}"))
    print(f"{name:<12} acc(b1)={a1 * 100:.2f}%  acc(b32)={a32 * 100:.2f}%  слои={layer_shapes(m)}")


# float32 это просто копия эталона
shutil.copy(BASELINE, os.path.join(MODELS_DIR, "float32.tt"))
m = fresh_model()
evaluate_and_save("float32", m, "fp32")

# честный эталон, то же дообучение что у прунинга, но без самого прунинга
m = fresh_model()
finetune(m, x_train, y_train, epochs=args.epochs)
evaluate_and_save("float32_ft3", m, "fp32")

# int8 берём из эталона без дообучения
m = fresh_model()
m.quant()
evaluate_and_save("int8", m, "int8_dyn")

# структурный прунинг от 10 до 70% и дообучение
for pct in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
    m = fresh_model()
    m.net.prune(amount=pct)  # режем нейроны Linear-слоёв, свёртки не трогаем
    finetune(m, x_train, y_train, epochs=args.epochs)  # optimizer создаётся уже после прунинга
    evaluate_and_save(f"prune{int(pct * 100)}", m, "fp32")

write_csv(os.path.join(SHARED, "accuracy.csv"),
          ["variant", "precision", "accuracy_b1", "accuracy_b32", "size_kb"], rows)
print(f"\nвсе версии в {MODELS_DIR}, точность в {os.path.join(SHARED, 'accuracy.csv')}")
