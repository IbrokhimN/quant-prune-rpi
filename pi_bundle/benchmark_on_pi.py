# бенчмарк numpy-бэкенда (tinytensor) на Raspberry Pi, только инференс.
#   python benchmark_on_pi.py                          # полный прогон
#   python benchmark_on_pi.py --quick                  # быстрая проверка, что всё запускается
#   OPENBLAS_NUM_THREADS=1 python benchmark_on_pi.py   # то же с одним потоком BLAS
#
# что меряем
#   latency на batch=1 и batch=32, чистый forward без DataLoader и softmax
#   несколько раундов со случайным порядком вариантов, итог это медиана по раундам и разброс между ними
#   accuracy на batch=1 и batch=32, у динамического int8 она зависит от батча
#   int8 в двух режимах, dynamic-range W8A8 и weight-only
#   температуру, частоту и троттлинг пишем в лог
import argparse
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (os.path.join(HERE, "..", "shared"), HERE):  # HERE идёт первым, там лежит tinytensor
    sys.path.insert(0, p)

import numpy as np                                         # noqa: E402
from tinytensor.models import LeNet                        # noqa: E402
from experiment import (accuracy_report, batched_predict, bench_loop, collect_env, count_params_any,
                        file_size_kb, first_existing, flops_any, gzip_size_kb, measure_ram_numpy,
                        numpy_forward_fn, numpy_legacy_predict_fn, pi_state, summarize_rounds,
                        throttled_now, write_results_csv)  # noqa: E402
from quant_pipelines import describe_pipeline, to_weight_only  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--models", default=first_existing(os.path.join(HERE, "models"),
                                                    os.path.join(HERE, "..", "shared", "models")))
ap.add_argument("--testset", default=first_existing(os.path.join(HERE, "testset.npz"),
                                                     os.path.join(HERE, "..", "shared", "testset.npz")))
ap.add_argument("--out", default=os.path.join(HERE, "results"))
ap.add_argument("--rounds", type=int, default=3)
ap.add_argument("--runs-b1", type=int, default=200)
ap.add_argument("--runs-b32", type=int, default=30)
ap.add_argument("--warmup", type=int, default=10)
ap.add_argument("--acc-n", type=int, default=10000, help="сколько тестовых объектов для accuracy")
ap.add_argument("--cooldown", type=float, default=5.0, help="пауза между раундами, с")
ap.add_argument("--quick", action="store_true")
args = ap.parse_args()
if args.quick:
    args.rounds, args.runs_b1, args.runs_b32, args.warmup, args.acc_n = 1, 30, 5, 3, 500

blas = os.environ.get("OPENBLAS_NUM_THREADS") or "auto"
tag = f"numpy_blas{blas}"

env = collect_env()
print("окружение:", json.dumps({k: env[k] for k in ("platform", "machine", "python", "python_bits",
                                                       "numpy", "cpu_model", "cpu_governor")}, ensure_ascii=False))
print("потоки BLAS:", blas, "| состояние:", env["pi_state_start"])
if env["cpu_governor"] not in (None, "performance"):
    print("!! governor =", env["cpu_governor"], "(частота будет плавать). Для стабильных замеров выполните\n"
          "   echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor")

data = np.load(args.testset)
x_test, y_test = data["x"][:args.acc_n], data["y"][:args.acc_n]

variants = []
for path in sorted(glob.glob(os.path.join(args.models, "*.tt"))):
    stem = os.path.basename(path)[:-3]
    m = LeNet(num_classes=10, in_channels=1)
    m.load(path)
    quantized = any(getattr(l, "w_q", None) is not None for l in m.net.layers)
    if quantized:
        variant = "float32" if stem == "int8" else stem.replace("_int8", "")
        modes = [("int8_dyn", "dynamic", m)]
        m_wo = LeNet(num_classes=10, in_channels=1)
        m_wo.load(path)
        modes.append(("int8_wo", "weight_only", to_weight_only(m_wo)))
    else:
        variant, modes = stem, [("fp32", None, m)]
    for precision, mode, model in modes:
        variants.append({"name": f"{variant}/{precision}", "variant": variant,
                         "precision": precision, "path": path, "model": model, "mode": mode})
if not variants:
    sys.exit(f"нет .tt моделей в {args.models}")

pipelines = {}
for v in variants:
    v["fn1"] = v["fn32"] = numpy_forward_fn(v["model"])
    v["legacy32"] = numpy_legacy_predict_fn(v["model"], 32)
    pipelines[v["name"]] = describe_pipeline(v["model"], v["mode"])
    v["pipeline"] = pipelines[v["name"]]["pipeline"]
print(f"\nвариантов: {len(variants)}")
for n, d in pipelines.items():
    if d["pipeline"] != "float32":
        print(f"  {n}: {d['pipeline']}")

# accuracy на batch=32 меряем на всём тесте, заодно это прогревает cpu. полный проход на batch=1 на Pi
# долгий, а нужен он только там, где предсказание зависит от батча (динамический int8, scale это
# max по батчу). у остальных проверяем на первых 500 объектах, что b1 и b32 дают одинаковые ответы,
# и если так, берём accuracy_b1 равной accuracy_b32. при любом расхождении гоним полный проход на
# batch=1. в колонке acc_b1_method в CSV видно, какой случай был
print("\n== accuracy ==")
K = min(500, len(x_test))
for v in variants:
    p32 = batched_predict(v["fn32"], x_test, 32)
    a32 = accuracy_report(p32, y_test)
    batch_dep = pipelines[v["name"]]["batch_dependent"]
    method = "measured"
    if not batch_dep:
        mism = int((batched_predict(v["fn1"], x_test[:K], 1) != p32[:K]).sum())
        if mism == 0:
            a1, method = a32, f"inferred (b1==b32 на {K} объектах)"
        else:
            batch_dep = True
            print(f"  !! {v['name']}, b1 != b32 на {mism} из {K} объектов, меряю b1 полностью")
    if batch_dep:
        a1 = accuracy_report(batched_predict(v["fn1"], x_test, 1), y_test)
    v["acc"], v["acc_method"] = (a1, a32), method
    print(f"  {v['name']:<26} b1={a1['accuracy'] * 100:.2f}% [{a1['acc_ci_lo'] * 100:.2f};{a1['acc_ci_hi'] * 100:.2f}]"
          f"  b32={a32['accuracy'] * 100:.2f}%  ({method})")

print("\n== latency ==")
bench_loop(variants, x_test, rounds=args.rounds, runs_b1=args.runs_b1, runs_b32=args.runs_b32,
           warmup=args.warmup, cooldown_s=args.cooldown)

rows = []
for v in variants:
    a1, a32 = v["acc"]
    r = {
        "runtime": "numpy", "threads": f"blas{blas}", "variant": v["variant"],
        "precision": v["precision"], "label": v["name"], "pipeline": v["pipeline"],
        "accuracy_b1": round(a1["accuracy"], 4), "acc_b1_ci_lo": round(a1["acc_ci_lo"], 4),
        "acc_b1_ci_hi": round(a1["acc_ci_hi"], 4), "accuracy_b32": round(a32["accuracy"], 4),
        "n_test": a1["n_test"], "acc_b1_method": v["acc_method"],
        "size_kb": round(file_size_kb(v["path"]), 1), "size_gzip_kb": round(gzip_size_kb(v["path"]), 1),
        "ram_mb": round(measure_ram_numpy(v["model"], x_test[:32]), 3),
        "flops": flops_any(v["model"]), "params": count_params_any(v["model"]),
    }
    r.update(summarize_rounds(v))
    rows.append(r)

os.makedirs(args.out, exist_ok=True)
csv_path = os.path.join(args.out, f"pi_{tag}.csv")
write_results_csv(csv_path, rows)
env["pi_state_end"] = pi_state()
with open(os.path.join(args.out, f"pi_env_{tag}.json"), "w") as f:
    json.dump({"env": env, "args": vars(args), "pipelines": pipelines}, f, indent=2, ensure_ascii=False)

print(f"\nготово, {csv_path}")
if any(r["throttled_any"] == 1 for r in rows):
    print("!! во время замеров был троттлинг или недобор питания, эти числа ненадёжны. "
          "Охлаждение, нормальный БП 5V/2.5A+, повторить.")
print("скопируй папку results/ на ПК в research/results/pi/ и запусти pc/03_plot_results.py")
