# общие функции замеров. их используют и на пк, и на распбери, и для numpy-бэкенда, и для litert,
# чтобы методика везде была одинаковая
import gc
import gzip
import os
import platform
import struct
import subprocess
import sys
import time
import tracemalloc

import numpy as np


def latency_stats(times_ms):
    t = np.asarray(times_ms, dtype=np.float64)
    return {
        "p50_ms": float(np.median(t)),
        "p95_ms": float(np.percentile(t, 95)),
        "mean_ms": float(t.mean()),
        "std_ms": float(t.std()),
        "min_ms": float(t.min()),
        "n": int(t.size),
    }


def time_calls(fn, inputs, runs, warmup):
    # inputs крутим по кругу, чтобы не мерить один и тот же батч из кэша
    for i in range(warmup):
        fn(inputs[i % len(inputs)])
    times = np.empty(runs, dtype=np.float64)
    gc_was = gc.isenabled()
    gc.collect()
    gc.disable()  # gc посреди замера даёт выброс
    try:
        for i in range(runs):
            x = inputs[i % len(inputs)]
            t0 = time.perf_counter_ns()
            fn(x)
            times[i] = (time.perf_counter_ns() - t0) / 1e6
    finally:
        if gc_was:
            gc.enable()
    return times


def make_batches(x, batch_size, n_batches):
    # contiguous делаем заранее, чтобы копирование не попало в замер
    n = len(x)
    out = []
    for i in range(n_batches):
        s = (i * batch_size) % max(1, n - batch_size)
        out.append(np.ascontiguousarray(x[s:s + batch_size]))
    return out


def numpy_forward_fn(model):
    # просто forward и argmax, softmax не нужен, argmax от него не меняется
    from tinytensor.core.tensor import Tensor
    model.eval()

    def fn(xb):
        return np.argmax(model(Tensor(xb)).data, axis=1)
    return fn


def numpy_legacy_predict_fn(model, batch_size):
    # старый способ через model.predict, оставлен для сверки со старой таблицей
    model.eval()

    def fn(xb):
        return model.predict(xb, batch_size=batch_size)
    return fn


def batched_predict(fn, x, bs):
    # хвост добиваем нулями до полного батча (на max|x| это не влияет, так что масштаб
    # динамического int8 не портится), лишние ответы потом отрезаем
    out = []
    n = len(x)
    for s in range(0, n, bs):
        xb = x[s:s + bs]
        k = len(xb)
        if k < bs:
            pad = np.zeros((bs - k,) + xb.shape[1:], dtype=xb.dtype)
            xb = np.concatenate([xb, pad])
        out.append(np.asarray(fn(np.ascontiguousarray(xb)))[:k])
    return np.concatenate(out)


def measure_ram_numpy(model, x_batch):
    # пик python-аллокаций за один forward, память на стороне C tracemalloc не видит.
    # для litert не подходит, там ram_mb остаётся пустым
    fn = numpy_forward_fn(model)
    tracemalloc.start()
    fn(x_batch)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak / (1024 ** 2)


def wilson_ci(k, n, z=1.96):
    # 95% доверительный интервал Вильсона для доли k/n
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (float(c - h), float(c + h))


def accuracy_report(preds, y_true):
    y = np.asarray(y_true).astype(np.int64).ravel()
    p = np.asarray(preds).astype(np.int64).ravel()
    k = int((p == y).sum())
    lo, hi = wilson_ci(k, len(y))
    return {"accuracy": k / len(y), "acc_ci_lo": lo, "acc_ci_hi": hi, "n_test": len(y)}


def file_size_kb(path):
    return os.path.getsize(path) / 1024.0


def gzip_size_kb(path):
    # размер после gzip, по нему видно, как сжимаются нули от неструктурного прунинга
    with open(path, "rb") as f:
        raw = f.read()
    return len(gzip.compress(raw, compresslevel=9)) / 1024.0


def _layer_weight_shape(layer):
    # форма весов и для float, и для int8-слоёв (у int8 float-весов нет)
    if getattr(layer, "weight", None) is not None:
        return tuple(layer.weight.data.shape)
    if getattr(layer, "w_q", None) is not None:
        return tuple(layer.w_q.shape)
    return None


def count_params_any(model):
    # у квантованных слоёв parameters() пустой, поэтому считаем по w_q
    total = 0
    for layer in model.net.layers:
        shp = _layer_weight_shape(layer)
        if shp is None:
            continue
        total += int(np.prod(shp))
        b = getattr(layer, "bias", None)
        b = b.data if b is not None else getattr(layer, "b_data", None)
        if b is not None:
            total += int(np.size(b))
    return total


def flops_any(model, input_shape=(1, 1, 28, 28)):
    # MAC'и на один объект по Conv2d и Linear. форму берём из реальных массивов, потому что
    # in_features и out_features после load() у прунинговых моделей устаревают
    from tinytensor.core.tensor import Tensor
    from tinytensor.nn.conv import Conv2d
    from tinytensor.nn.linear import Linear
    x = Tensor(np.zeros(input_shape, dtype=np.float32))
    total = 0
    for layer in model.net.layers:
        x = layer(x)
        if isinstance(layer, Conv2d):
            _, oc, oh, ow = x.data.shape
            total += oh * ow * oc * layer.in_channels * layer.kh * layer.kw
        elif isinstance(layer, Linear):
            shp = _layer_weight_shape(layer)
            total += shp[0] * shp[1]
    return int(total)


def layer_shapes(model):
    # для логов, какие слои реально порезал прунинг
    return [list(_layer_weight_shape(l)) for l in model.net.layers if _layer_weight_shape(l)]


def _read(path):
    try:
        with open(path) as f:
            return f.read().strip()
    except Exception:
        return None


def _vcgencmd(*args):
    try:
        return subprocess.run(["vcgencmd", *args], capture_output=True, text=True,
                              timeout=2).stdout.strip()
    except Exception:
        return None


def pi_state():
    # температура, частота и флаги троттлинга. вне Pi все поля будут None
    s = {"temp_c": None, "freq_mhz": None, "throttled": None}
    t = _read("/sys/class/thermal/thermal_zone0/temp")
    if t:
        s["temp_c"] = round(int(t) / 1000.0, 1)
    f = _read("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq")
    if f:
        s["freq_mhz"] = round(int(f) / 1000.0)
    th = _vcgencmd("get_throttled")  # вида throttled=0x50005
    if th and "=" in th:
        s["throttled"] = th.split("=")[1]
    return s


def throttled_now(hex_str):
    # биты 0..3 это проблемы прямо сейчас (undervolt, freq cap, throttled, soft temp limit)
    if not hex_str:
        return None
    return bool(int(hex_str, 16) & 0xF)


def collect_env():
    env = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version.split()[0],
        "python_bits": struct.calcsize("P") * 8,
        "numpy": np.__version__,
        "board": (_read("/proc/device-tree/model") or "").replace("\x00", "") or None,
        "cpu_model": None,
        "cpu_governor": _read("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"),
        "cpu_max_mhz": None,
        "n_cpus": os.cpu_count(),
        "env_threads": {k: os.environ.get(k) for k in
                        ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")},
        "pi_state_start": pi_state(),
    }
    ci = _read("/proc/cpuinfo") or ""
    for line in ci.splitlines():
        if line.lower().startswith(("model name", "hardware")):
            env["cpu_model"] = line.split(":", 1)[1].strip()
            break
    mf = _read("/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq")
    if mf:
        env["cpu_max_mhz"] = round(int(mf) / 1000.0)
    try:
        cfg = np.show_config(mode="dicts")
        env["numpy_blas"] = cfg.get("Build Dependencies", {}).get("blas", {})
    except Exception:
        env["numpy_blas"] = None
    return env


def bench_loop(variants, x_test, rounds=3, runs_b1=200, runs_b32=30, warmup=10,
               cooldown_s=5.0, n_inputs=64, legacy_runs=10, seed=0, log=print):
    # variants это список словарей с name, fn1 (batch=1), fn32 (batch=32) и необязательным legacy32.
    # замеры идут несколькими раундами, порядок вариантов в каждом раунде случайный, чтобы дрейф
    # температуры и частоты не бил всегда по одним и тем же. итог это медиана медиан и разброс
    # между раундами
    b1 = make_batches(x_test, 1, n_inputs)
    b32 = make_batches(x_test, 32, max(8, n_inputs // 8))
    rng = np.random.RandomState(seed)
    for v in variants:
        v["rounds"] = {"b1": [], "b32": [], "legacy32": [], "temp": [], "throttled": []}

    for r in range(rounds):
        order = rng.permutation(len(variants))
        for vi in order:
            v = variants[vi]
            s0 = pi_state()
            st1 = latency_stats(time_calls(v["fn1"], b1, runs_b1, warmup))
            st32 = latency_stats(time_calls(v["fn32"], b32, runs_b32, max(3, warmup // 2)))
            v["rounds"]["b1"].append(st1)
            v["rounds"]["b32"].append(st32)
            if v.get("legacy32") is not None:
                lg = latency_stats(time_calls(v["legacy32"], b32[:1], legacy_runs, 2))
                v["rounds"]["legacy32"].append(lg)
            s1 = pi_state()
            for s_ in (s0, s1):
                if s_["temp_c"] is not None:
                    v["rounds"]["temp"].append(s_["temp_c"])
                thr = throttled_now(s_["throttled"])
                if thr is not None:
                    v["rounds"]["throttled"].append(thr)
            temp = s1["temp_c"]
            log(f"[раунд {r + 1}/{rounds}] {v['name']:<28} b1 p50={st1['p50_ms']:.3f}мс "
                f"p95={st1['p95_ms']:.3f} | b32 p50={st32['p50_ms']:.2f}мс"
                + (f" | {temp}°C" if temp is not None else "")
                + (" | !!THROTTLED" if throttled_now(s1["throttled"]) else ""))
        if r < rounds - 1 and cooldown_s > 0:
            time.sleep(cooldown_s)


def summarize_rounds(v):
    def agg(key):
        p50 = np.array([s["p50_ms"] for s in v["rounds"][key]])
        p95 = np.array([s["p95_ms"] for s in v["rounds"][key]])
        med = float(np.median(p50))
        spread = float((p50.max() - p50.min()) / med) if med > 0 and len(p50) > 1 else 0.0
        return med, spread, float(np.median(p95)), float(np.mean([s["std_ms"] for s in v["rounds"][key]]))
    b1, b1s, b1p95, b1std = agg("b1")
    b32, b32s, b32p95, b32std = agg("b32")
    out = {
        "lat_b1_ms": round(b1, 4), "lat_b1_ms_spread": round(b1s, 4),
        "lat_b1_p95_ms": round(b1p95, 4), "lat_b1_std_ms": round(b1std, 4),
        "lat_b32_ms": round(b32, 3), "lat_b32_ms_spread": round(b32s, 4),
        "lat_b32_p95_ms": round(b32p95, 3), "lat_b32_std_ms": round(b32std, 3),
        "ms_per_img_b32": round(b32 / 32.0, 4),
        "throughput_ips_b32": round(32.0 / (b32 / 1000.0), 1),
        "legacy_predict_b32_ms": "",
        "temp_c_max": max(v["rounds"]["temp"]) if v["rounds"]["temp"] else "",
        "throttled_any": int(any(v["rounds"]["throttled"])) if v["rounds"]["throttled"] else "",
    }
    if v["rounds"]["legacy32"]:
        out["legacy_predict_b32_ms"] = round(float(np.median(
            [s["p50_ms"] for s in v["rounds"]["legacy32"]])), 3)
    return out


RESULT_COLUMNS = [
    "runtime", "threads", "variant", "precision", "label", "pipeline",
    "accuracy_b1", "acc_b1_ci_lo", "acc_b1_ci_hi", "accuracy_b32", "n_test", "acc_b1_method",
    "lat_b1_ms", "lat_b1_ms_spread", "lat_b1_p95_ms", "lat_b1_std_ms",
    "lat_b32_ms", "lat_b32_ms_spread", "lat_b32_p95_ms", "lat_b32_std_ms",
    "ms_per_img_b32", "throughput_ips_b32", "legacy_predict_b32_ms",
    "size_kb", "size_gzip_kb", "ram_mb", "flops", "params", "temp_c_max", "throttled_any",
]


def write_results_csv(path, rows):
    import csv
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RESULT_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in RESULT_COLUMNS})


def first_existing(*paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return paths[0]
