# собирает одну папку и архив для копирования на Raspberry Pi, внутри код, модели, тест и tflite.
# так не надо вручную копировать пять разных вещей.
#   python 06_make_pi_bundle.py  даёт ../pi_bundle/ и ../pi_bundle.zip
# на Pi дальше хватит unzip pi_bundle.zip && cd pi_bundle && python benchmark_on_pi.py
import glob
import os
import shutil

from common import HERE, SHARED, MODELS_DIR, TESTSET

ROOT = os.path.abspath(os.path.join(HERE, ".."))
OUT = os.path.join(ROOT, "pi_bundle")
if os.path.exists(OUT):
    shutil.rmtree(OUT)
os.makedirs(OUT)

ignore = shutil.ignore_patterns("__pycache__", "*.pyc", "*.so", "cuda_gpu.*", "cuda_binding.cpp")
shutil.copytree(os.path.join(ROOT, "raspberry_pi", "tinytensor"), os.path.join(OUT, "tinytensor"),
                ignore=ignore)

for f in ("benchmark_on_pi.py", "benchmark_litert.py"):
    shutil.copy(os.path.join(ROOT, "raspberry_pi", f), OUT)
for f in ("experiment.py", "quant_pipelines.py", "litert_utils.py", "unstructured.py"):
    shutil.copy(os.path.join(SHARED, f), OUT)

shutil.copytree(MODELS_DIR, os.path.join(OUT, "models"))
shutil.copy(TESTSET, os.path.join(OUT, "testset.npz"))
tfl = os.path.join(SHARED, "tflite")
if os.path.isdir(tfl) and glob.glob(os.path.join(tfl, "*.tflite")):
    shutil.copytree(tfl, os.path.join(OUT, "tflite"))
else:
    print("!! нет ../shared/tflite, запусти 05_export_tflite.py, иначе benchmark_litert.py не запустится")

zip_path = shutil.make_archive(OUT, "zip", ROOT, "pi_bundle")
print(f"готово, {OUT}\n{zip_path} ({os.path.getsize(zip_path) / 1024:.0f} КБ)")
