# бенчмарк тех же весов в LiteRT (бывший TFLite) на Raspberry Pi. нужен, чтобы понять, какие выводы
# про numpy-бэкенд относятся к самой идее (прунинг, квантизация), а какие к наивной реализации.
# методика замеров та же, shared/experiment.py.
#   python benchmark_litert.py                 # threads 1 и 4
#   python benchmark_litert.py --threads 1
#   python benchmark_litert.py --quick
#
# ставится через pip install ai-edge-litert, на Pi для этого нужна 64-bit ОС (см. README)
import argparse
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (os.path.join(HERE, "..", "shared"), HERE):
    sys.path.insert(0, p)

import numpy as np                                         # noqa: E402
from experiment import (accuracy_report, batched_predict, bench_loop, collect_env, file_size_kb,
                        first_existing, gzip_size_kb, pi_state, summarize_rounds,
                        write_results_csv)                 # noqa: E402
from litert_utils import make_litert_fn, runtime_info      # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--tflite", default=first_existing(os.path.join(HERE, "tflite"),
                                                    os.path.join(HERE, "..", "shared", "tflite")))
ap.add_argument("--models", default=first_existing(os.path.join(HERE, "models"),
                                                    os.path.join(HERE, "..", "shared", "models")),
                help="нужна только чтобы посчитать flops/params (опционально)")
ap.add_argument("--testset", default=first_existing(os.path.join(HERE, "testset.npz"),
                                                     os.path.join(HERE, "..", "shared", "testset.npz")))
ap.add_argument("--out", default=os.path.join(HERE, "results"))
ap.add_argument("--threads", type=int, nargs="+", default=[1, 4])
ap.add_argument("--rounds", type=int, default=3)
ap.add_argument("--runs-b1", type=int, default=500)
ap.add_argument("--runs-b32", type=int, default=100)
ap.add_argument("--warmup", type=int, default=20)
ap.add_argument("--acc-n", type=int, default=10000)
ap.add_argument("--cooldown", type=float, default=5.0)
ap.add_argument("--quick", action="store_true")
args = ap.parse_args()
if args.quick:
    args.rounds, args.runs_b1, args.runs_b32, args.warmup, args.acc_n = 1, 50, 10, 5, 500

env = collect_env()
env.update(runtime_info())
print("рантайм:", runtime_info(), "| платформа:", env["platform"], env["machine"])
if env["cpu_governor"] not in (None, "performance"):
    print("!! governor =", env["cpu_governor"], "(для стабильных замеров лучше performance, см. README)")

data = np.load(args.testset)
x_test, y_test = data["x"][:args.acc_n], data["y"][:args.acc_n]

# flops/params берём из .tt (те же архитектуры), если tinytensor и модели рядом
try:
    from tinytensor.models import LeNet
    from experiment import flops_any, count_params_any
    have_tt = True
except Exception:
    have_tt = False

files = sorted(glob.glob(os.path.join(args.tflite, "*.tflite")))
if not files:
    sys.exit(f"нет .tflite в {args.tflite} (сначала pc/05_export_tflite.py)")

for threads in args.threads:
    print(f"\n== LiteRT, num_threads={threads} ==")
    variants = []
    for path in files:
        base = os.path.basename(path)[:-7]
        variant, precision = base.split("__")
        v = {"name": f"{variant}/{precision}", "variant": variant, "precision": precision, "path": path}
        v["fn1"] = make_litert_fn(path, 1, threads)
        v["fn32"] = make_litert_fn(path, 32, threads)
        v["legacy32"] = None
        variants.append(v)

    print("== accuracy ==")
    for v in variants:
        a1 = accuracy_report(batched_predict(v["fn1"], x_test, 1), y_test)
        a32 = accuracy_report(batched_predict(v["fn32"], x_test, 32), y_test)
        v["acc"] = (a1, a32)
        print(f"  {v['name']:<26} b1={a1['accuracy'] * 100:.2f}%  b32={a32['accuracy'] * 100:.2f}%")

    print("== latency ==")
    bench_loop(variants, x_test, rounds=args.rounds, runs_b1=args.runs_b1, runs_b32=args.runs_b32,
               warmup=args.warmup, cooldown_s=args.cooldown, n_inputs=64)

    rows = []
    for v in variants:
        a1, a32 = v["acc"]
        r = {
            "runtime": "litert", "threads": f"t{threads}", "variant": v["variant"],
            "precision": v["precision"], "label": v["name"],
            "pipeline": {"fp32": "litert float32",
                         "int8_dyn": "litert dynamic-range (int8 weights, float activations)",
                         "int8_full": "litert full-integer (static int8, float I/O)"}[v["precision"]],
            "accuracy_b1": round(a1["accuracy"], 4), "acc_b1_ci_lo": round(a1["acc_ci_lo"], 4),
            "acc_b1_ci_hi": round(a1["acc_ci_hi"], 4), "accuracy_b32": round(a32["accuracy"], 4),
            "n_test": a1["n_test"], "acc_b1_method": "measured",
            "size_kb": round(file_size_kb(v["path"]), 1), "size_gzip_kb": round(gzip_size_kb(v["path"]), 1),
            "ram_mb": "",  # tracemalloc не видит C++-аллокации рантайма, сравнивать нечего
        }
        tt_path = os.path.join(args.models, v["variant"] + ".tt")
        if have_tt and os.path.exists(tt_path):
            m = LeNet(num_classes=10, in_channels=1)
            m.load(tt_path)
            r["flops"], r["params"] = flops_any(m), count_params_any(m)
        r.update(summarize_rounds(v))
        rows.append(r)

    os.makedirs(args.out, exist_ok=True)
    tag = f"litert_t{threads}"
    csv_path = os.path.join(args.out, f"pi_{tag}.csv")
    write_results_csv(csv_path, rows)
    env["pi_state_end"] = pi_state()
    with open(os.path.join(args.out, f"pi_env_{tag}.json"), "w") as f:
        json.dump({"env": env, "args": vars(args)}, f, indent=2, ensure_ascii=False)
    print(f"готово, {csv_path}")
    if any(r["throttled_any"] == 1 for r in rows):
        print("!! был троттлинг/недобор питания - числа ненадёжны")
