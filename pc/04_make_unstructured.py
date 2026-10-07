# неструктурный прунинг. зануляем самые малые по модулю веса Linear-слоёв, форму матриц не меняем.
# на выходе ../shared/models/unstruct{10..70}.tt и ../shared/accuracy_unstruct.csv
# маска держится всё дообучение, а sparsity считаем и до него, и после, и сверяем.
# запускать после 01_train_baseline.py
#   python 04_make_unstructured.py [--epochs 3]
import argparse
import os

from common import (MODELS_DIR, SHARED, USE_CUDA, fresh_model, finetune, linear_masks, load_data,
                    write_csv, FINETUNE_EPOCHS)
from experiment import numpy_forward_fn, batched_predict, accuracy_report, file_size_kb
from unstructured import prune_unstructured, sparsity

ap = argparse.ArgumentParser()
ap.add_argument("--epochs", type=int, default=FINETUNE_EPOCHS)
args = ap.parse_args()
print(f"дообучение на {'cuda' if USE_CUDA else 'cpu'}, эпох {args.epochs}")

(x_train, y_train), (x_test, y_test) = load_data()
os.makedirs(MODELS_DIR, exist_ok=True)
rows = []

for pct in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
    m = fresh_model()
    prune_unstructured(m, amount=pct)
    sp_before = sparsity(m)
    finetune(m, x_train, y_train, epochs=args.epochs, masks=linear_masks(m))
    sp_after = sparsity(m)
    assert sp_after >= sp_before - 1e-6, "маска не удержалась, разреженность упала"

    fn = numpy_forward_fn(m)
    a1 = accuracy_report(batched_predict(fn, x_test, 1), y_test)["accuracy"]
    a32 = accuracy_report(batched_predict(fn, x_test, 32), y_test)["accuracy"]
    name = f"unstruct{int(pct * 100)}"
    path = os.path.join(MODELS_DIR, f"{name}.tt")
    m.save(path)
    rows.append((name, "fp32", f"{a1:.4f}", f"{a32:.4f}", f"{sp_before:.4f}", f"{sp_after:.4f}",
                 f"{file_size_kb(path):.1f}"))
    print(f"{name}: acc(b1)={a1 * 100:.2f}%  acc(b32)={a32 * 100:.2f}%  "
          f"разреженность до/после дообучения = {sp_before * 100:.0f}% / {sp_after * 100:.0f}%")

write_csv(os.path.join(SHARED, "accuracy_unstruct.csv"),
          ["variant", "precision", "accuracy_b1", "accuracy_b32", "sparsity_before",
           "sparsity_after", "size_kb"], rows)
print(f"\nнеструктурные версии в {MODELS_DIR}")
