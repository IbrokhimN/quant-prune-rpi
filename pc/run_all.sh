#!/usr/bin/env bash
# весь пк-конвейер одной командой. baseline.tt не перезаписывается, шаг 01 запускайте вручную
set -euo pipefail
cd "$(dirname "$0")"
[ -f ../shared/baseline.tt ] || python 01_train_baseline.py
python 00_prepare_testset.py
python 02_make_variants.py
python 04_make_unstructured.py
python 05_export_tflite.py
python 06_make_pi_bundle.py
echo "готово, теперь скопируйте ../pi_bundle.zip на Pi (см. README)"
