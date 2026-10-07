# анализ результатов с Pi, на выходе таблица, парето-отчёт и три картинки на каждую группу замеров.
#   python 03_plot_results.py [--results ../results] [--acc-tol 0.002]
# читает все results/**/pi_*.csv, их делают benchmark_on_pi.py и benchmark_litert.py.
# в --out (по умолчанию ../results/analysis) пишет
#   table1.md и table1.csv          сводная таблица
#   pareto_report.md                фронты (строгий, робастный, основной) и кто кого доминирует
#   pareto_membership.csv           то же, но для машин
#   <группа>/fig1_tradeoff.png      сжатие против ускорения, главный рисунок статьи
#   <группа>/fig2_accuracy_ci.png   точность с доверительными интервалами
#   <группа>/fig3_pareto.png        парето-фронт размер на latency (b=1 и b=32)
#   all_runtimes_lat_b1.png         только если в results больше одного рантайма
import argparse
import csv
import glob
import math
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from common import HERE, SHARED  # noqa: F401 (нужен, чтобы ../shared попал в sys.path)
from pareto import analyse, family, non_dominated, Obj

ap = argparse.ArgumentParser()
ap.add_argument("--results", default=os.path.join(HERE, "..", "results"))
ap.add_argument("--out", default=None)
ap.add_argument("--acc-tol", type=float, default=0.002,
                help="допуск accuracy в робастном режиме, доля. 0.002 это 0.2 п.п., примерно полуширина 95%% ДИ на 10k")
ap.add_argument("--lat-tol", type=float, default=0.03, help="относительный допуск latency")
args = ap.parse_args()
OUT = args.out or os.path.join(args.results, "analysis")
os.makedirs(OUT, exist_ok=True)
for old in glob.glob(os.path.join(OUT, "pareto_*.png")):  # картинки старого формата
    os.remove(old)

NUM = ["accuracy_b1", "acc_b1_ci_lo", "acc_b1_ci_hi", "accuracy_b32", "lat_b1_ms", "lat_b1_ms_spread",
       "lat_b1_p95_ms", "lat_b32_ms", "lat_b32_ms_spread", "lat_b32_p95_ms", "ms_per_img_b32",
       "throughput_ips_b32", "legacy_predict_b32_ms", "size_kb", "size_gzip_kb", "ram_mb", "flops",
       "params", "temp_c_max", "throttled_any"]

plt.rcParams.update({"font.size": 11, "axes.titlesize": 13, "axes.labelsize": 11.5,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "grid.linestyle": "--"})

COL = {"baseline": "#333333", "structured": "#1b9e77", "unstructured": "#7570b3",
       "int8-dyn": "#d95f02", "int8-wo": "#e6ab02", "int8-full": "#a6761d", "structured+int8": "#e7298a"}
MRK = {"baseline": "D", "structured": "o", "unstructured": "^", "int8-dyn": "s", "int8-wo": "s",
       "int8-full": "s", "structured+int8": "P"}
FAM_NAME = {"baseline": "эталон (float32)", "structured": "структурный прунинг", "unstructured": "неструктурный прунинг",
            "int8-dyn": "INT8 dynamic (W8A8)", "int8-wo": "INT8 weight-only", "int8-full": "INT8 full-integer",
            "structured+int8": "структурный + INT8"}
ORDER = list(COL)


def load_rows():
    rows = []
    files = sorted(glob.glob(os.path.join(args.results, "**", "pi_*.csv"), recursive=True))
    if not files:
        raise SystemExit(f"нет pi_*.csv в {args.results}. Скопируй results/ с Pi (см. README).")
    for fpath in files:
        with open(fpath, newline="") as f:
            for r in csv.DictReader(f):
                for k in NUM:
                    try:
                        r[k] = float(r[k])
                    except (KeyError, ValueError):
                        r[k] = None
                r["group"] = f"{r['runtime']} {r['threads']}"
                r["family"] = family(r["variant"], r["precision"])
                m = re.search(r"(\d+)$", r["variant"])
                r["pct"] = int(m.group(1)) if m else 0
                rows.append(r)
    return rows


def by_label(rs):
    return {r["label"]: r for r in rs}


def slug(g):
    return g.replace(" ", "_")


def sort_key(r):
    return (ORDER.index(r["family"]) if r["family"] in ORDER else 99, r["pct"], r["label"])


rows = load_rows()
groups = sorted({r["group"] for r in rows})
runtimes = sorted({r["runtime"] for r in rows})
print("группы:", groups, "| строк:", len(rows))
if any(r["throttled_any"] == 1.0 for r in rows):
    print("!! в данных есть throttled_any=1 (недобор питания или троттлинг), смотри раздел ограничений в README")


def make_table():
    lines = ["| группа | вариант | acc b1, % [95% ДИ] | acc b32, % | lat b1 p50 (p95), мс | lat b32, мс | мс/объект (b32) "
             "| размер, КБ | gzip, КБ | MACs | ускорение b1 | ускорение b32 | сжатие |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    out = []
    f = lambda v, d=2: "-" if v is None else f"{v:.{d}f}"
    for g in groups:
        rs = sorted([r for r in rows if r["group"] == g], key=sort_key)
        base = by_label(rs).get("float32/fp32")
        for r in rs:
            sp1 = base["lat_b1_ms"] / r["lat_b1_ms"] if base and r["lat_b1_ms"] else None
            sp32 = base["lat_b32_ms"] / r["lat_b32_ms"] if base and r["lat_b32_ms"] else None
            comp = base["size_kb"] / r["size_kb"] if base and r["size_kb"] else None
            lines.append(
                f"| {g} | {r['label']} | {f(r['accuracy_b1'] * 100)} [{f(r['acc_b1_ci_lo'] * 100)}; "
                f"{f(r['acc_b1_ci_hi'] * 100)}] | {f(r['accuracy_b32'] * 100)} | {f(r['lat_b1_ms'], 3)} "
                f"({f(r['lat_b1_p95_ms'], 3)}) | {f(r['lat_b32_ms'], 2)} | {f(r['ms_per_img_b32'], 3)} | "
                f"{f(r['size_kb'], 1)} | {f(r['size_gzip_kb'], 1)} | {f(r['flops'], 0)} | "
                f"{f(sp1)}× | {f(sp32)}× | {f(comp)}× |")
            out.append([g, r["label"], r["accuracy_b1"], r["accuracy_b32"], r["lat_b1_ms"],
                        r["lat_b32_ms"], r["size_kb"], r["size_gzip_kb"], r["flops"], sp1, sp32, comp])
    with open(os.path.join(OUT, "table1.md"), "w") as fh:
        fh.write("Ускорение и сжатие считаются относительно float32/fp32 в той же группе (тот же рантайм и потоки).\n\n")
        fh.write("\n".join(lines) + "\n")
    with open(os.path.join(OUT, "table1.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["group", "label", "accuracy_b1", "accuracy_b32", "lat_b1_ms", "lat_b32_ms", "size_kb",
                    "size_gzip_kb", "macs", "speedup_b1", "speedup_b32", "compression"])
        w.writerows(out)


def acc_equivalent(rs):
    # если у 95% ДИ всех вариантов есть общее пересечение (в 1D это то же, что попарное перекрытие),
    # различия точности статистически не значимы и точность не должна разводить точки по фронту
    lo = max(r["acc_b1_ci_lo"] for r in rs)
    hi = min(r["acc_b1_ci_hi"] for r in rs)
    return lo <= hi, lo, hi


def main_front(rs, latcol):
    # основной фронт по размеру и latency, оба минимизируем. точность входит в критерии только если
    # её различия значимы, иначе все варианты по ДИ эквивалентны и точность просто не учитываем
    eq, lo, hi = acc_equivalent(rs)
    objs = [Obj("size_kb", "min"), Obj(latcol, "min", rel_tol=args.lat_tol, spread_col=latcol + "_spread")]
    if not eq:
        objs.append(Obj("accuracy_b1", "max", abs_tol=args.acc_tol))
    ok = [r for r in rs if all(isinstance(r.get(o.col), (int, float)) for o in objs)]
    front, dom = non_dominated(ok, objs)
    return ok, set(front), dom, eq, lo, hi


membership = []
report = ["# Парето-анализ\n",
          "Фронтов три. Основной (рисунок fig3) строится по размеру и latency при статистически "
          "неразличимой точности (95% ДИ перекрываются), его и стоит цитировать в статье. "
          "Строгий и робастный берут точность третьим критерием. Строгий сравнивает точные числа, "
          f"в робастном ничьёй считается разница accuracy меньше {args.acc_tol * 100:.1f} п.п. "
          f"или latency меньше max({args.lat_tol * 100:.0f}%, разброс между раундами). "
          "Строгий фронт по точности реагирует на шум в единицы объектов из 10 000.\n"]


def report_group(g, rs):
    report.append(f"\n## {g}\n")
    for latcol, name in (("lat_b1_ms", "batch=1"), ("lat_b32_ms", "batch=32")):
        ok, front, dom, eq, lo, hi = main_front(rs, latcol)
        report.append(f"\n**Основной фронт, размер x latency ({name})**. "
                      f"Точность, 95% ДИ {'перекрываются' if eq else 'не перекрываются'} "
                      f"(max нижней границы {lo * 100:.2f}, min верхней {hi * 100:.2f})\n")
        report.append("| вариант | на фронте | кто доминирует |")
        report.append("|---|---|---|")
        for i, r in sorted(enumerate(ok), key=lambda t: t[1]["size_kb"]):
            d = ", ".join(ok[j]["label"] for j in dom.get(i, [])[:3]) or "-"
            report.append(f"| {r['label']} | {'**да**' if i in front else 'нет'} | {d} |")
            membership.append([g, f"main_{latcol}", r["label"], int(i in front), "", d])
    for plane, title in (("size", "размер vs accuracy"), ("lat_b1", "latency b=1 vs accuracy"),
                         ("lat_b32", "latency b=32 vs accuracy")):
        s_ok, s_front, s_dom, _ = analyse(rs, plane, robust=False)
        r_ok, r_front, r_dom, _ = analyse(rs, plane, robust=True, acc_tol=args.acc_tol, lat_tol_rel=args.lat_tol)
        fam_on = {}
        for i in s_front:
            fam_on.setdefault(s_ok[i]["family"], []).append(s_ok[i]["label"])
        report.append(f"\n**{title}**. Строгий фронт, " + "; ".join(f"{k} ({', '.join(v)})" for k, v in sorted(fam_on.items()))
                      + f"\n\nРобастный фронт (допуск {args.acc_tol * 100:.1f} п.п.), "
                      + ", ".join(sorted(r_ok[i]["label"] for i in r_front)) + "\n")
        for i, r in enumerate(s_ok):
            membership.append([g, plane, r["label"], int(i in s_front), int(i in set(r_front)),
                               "; ".join(s_ok[j]["label"] for j in s_dom.get(i, []))])


def cluster_label(rs_):
    # одна подпись на кучу совпавших точек (float32, ft3, unstruct10-70 лежат в одной точке)
    parts = []
    base = sorted(r["variant"] for r in rs_ if r["family"] == "baseline")
    if base:
        parts.append(" · ".join(base))
    un = sorted(r["pct"] for r in rs_ if r["family"] == "unstructured")
    if un:
        parts.append(f"unstruct{un[0]}-{un[-1]}" if len(un) > 1 else f"unstruct{un[0]}")
    other = [r["label"] for r in rs_ if r["family"] not in ("baseline", "unstructured")]
    return "\n".join(parts + other)


def fig1_tradeoff(g, rs):
    base = by_label(rs).get("float32/fp32")
    if not base:
        return
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.6))
    for ax, (latcol, title) in zip(axes, (("lat_b1_ms", "batch = 1  (задержка одного запроса)"),
                                          ("lat_b32_ms", "batch = 32  (пропускная способность)"))):
        pt = lambda r: (base["size_kb"] / r["size_kb"], base[latcol] / r[latcol])
        ys = [pt(r)[1] for r in rs]
        ymin, ymax = min(min(ys), 1) - 0.08, max(max(ys), 1) + 0.09
        ax.axhspan(ymin, 1, color="#d95f02", alpha=0.06, lw=0)
        ax.axhspan(1, ymax, color="#1b9e77", alpha=0.06, lw=0)
        ax.axhline(1, color="#555", lw=1.2)
        ax.text(0.015, 0.97, "▲ быстрее эталона", transform=ax.transAxes, ha="left", va="top",
                fontsize=9.5, color="#1b7f62")
        ax.text(0.015, 0.03, "▼ медленнее эталона", transform=ax.transAxes, ha="left", va="bottom",
                fontsize=9.5, color="#b04a00")
        # структурный прунинг рисуем линией по степеням
        st = sorted([r for r in rs if r["family"] == "structured"], key=lambda r: r["pct"])
        if st:
            ax.plot([pt(r)[0] for r in st], [pt(r)[1] for r in st], "-o", color=COL["structured"], lw=2,
                    ms=7, zorder=4)
            for r in (st[0], st[len(st) // 2], st[-1]):
                x, y = pt(r)
                ax.annotate(f"prune{r['pct']}", (x, y), textcoords="offset points", xytext=(0, 11),
                            ha="center", fontsize=9.5, color=COL["structured"], fontweight="bold")
        # в точке (1, 1) лежат эталоны и неструктурный прунинг
        cl = [r for r in rs if r["family"] in ("baseline", "unstructured")]
        for r in cl:
            x, y = pt(r)
            ax.scatter(x, y, s=95 if r["family"] == "baseline" else 70, marker=MRK[r["family"]],
                       color=COL[r["family"]], alpha=0.85 if r["family"] == "baseline" else 0.45,
                       edgecolors="white", linewidths=0.6, zorder=5)
        un = [r for r in rs if r["family"] == "unstructured"]
        gz = max((base["size_gzip_kb"] / r["size_gzip_kb"]) for r in un) if un and all(r["size_gzip_kb"] for r in un) else None
        ax.annotate(cluster_label(cl) + (f"\n(gzip до {gz:.1f}×, но файл не меньше)" if gz else ""), pt(base),
                    textcoords="offset points", xytext=(18, -44 if latcol == "lat_b1_ms" else -50),
                    fontsize=9.5, color="#444", va="top",
                    arrowprops=dict(arrowstyle="-", color="#888", lw=0.8))
        # int8
        for r in rs:
            if r["family"].startswith("int8"):
                x, y = pt(r)
                ax.scatter(x, y, s=120, marker="s", color=COL[r["family"]], edgecolors="white", zorder=6)
                nm = {"int8-dyn": "INT8 dynamic", "int8-wo": "INT8 weight-only"}.get(r["family"], r["family"])
                ax.annotate(f"{nm}\n{y:.2f}×", (x, y), textcoords="offset points", xytext=(-10, 0),
                            ha="right", va="center", fontsize=9.5, color=COL[r["family"]], fontweight="bold")
        ax.set_xlim(0.7, max(pt(r)[0] for r in rs) * 1.12)
        ax.set_ylim(ymin, ymax)
        ax.set_xlabel("сжатие размера файла, ×   (правее = меньше файл)")
        ax.set_ylabel("ускорение относительно float32, ×   (выше = быстрее)")
        ax.set_title(title, loc="left")
    fig.suptitle(f"Сжатие ≠ ускорение  ·  {g}", x=0.01, ha="left", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fn = os.path.join(OUT, slug(g), "fig1_tradeoff.png")
    fig.savefig(fn, dpi=150)
    plt.close(fig)
    print("график:", fn)


def fig2_accuracy(g, rs):
    rs = sorted(rs, key=sort_key)
    ref = by_label(rs).get("float32_ft3/fp32") or by_label(rs).get("float32/fp32")
    fig, ax = plt.subplots(figsize=(9.5, 0.36 * len(rs) + 2.2))
    ypos = list(range(len(rs)))[::-1]
    if ref:
        ax.axvspan(ref["acc_b1_ci_lo"] * 100, ref["acc_b1_ci_hi"] * 100, color="#999", alpha=0.15, lw=0)
        ax.axvline(ref["accuracy_b1"] * 100, color="#555", lw=1, ls="--")
    prev = None
    for y, r in zip(ypos, rs):
        if prev is not None and r["family"] != prev:
            ax.axhline(y + 0.5, color="#ddd", lw=0.8)
        prev = r["family"]
        a = r["accuracy_b1"] * 100
        ax.errorbar(a, y, xerr=[[a - r["acc_b1_ci_lo"] * 100], [r["acc_b1_ci_hi"] * 100 - a]], fmt=MRK[r["family"]],
                    color=COL[r["family"]], ecolor=COL[r["family"]], elinewidth=1.8, capsize=3, ms=7)
        ax.text(r["acc_b1_ci_hi"] * 100 + 0.02, y, f"{a:.2f}", va="center", fontsize=8.5, color="#555")
    ax.set_yticks(ypos)
    ax.set_yticklabels([r["label"] for r in rs], fontsize=9.5)
    ax.set_xlabel("accuracy (batch = 1), %   ·   отрезок = 95% доверительный интервал Вильсона")
    ax.grid(axis="y", visible=False)
    if ref:
        ax.text(0.995, 0.005, f"серая полоса и пунктир это {ref['label']}", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=8.5, color="#777")
    eq, lo, hi = acc_equivalent(rs)
    ax.set_title(f"Точность  ·  {g}\n" + ("все интервалы перекрываются, различия точности не значимы"
                                            if eq else "внимание, не все интервалы перекрываются"),
                 loc="left", fontsize=12)
    fig.tight_layout()
    fn = os.path.join(OUT, slug(g), "fig2_accuracy_ci.png")
    fig.savefig(fn, dpi=150)
    plt.close(fig)
    print("график:", fn)


def draw_pareto(ax, rs, latcol, clip_top=None):
    ok, front, dom, eq, lo, hi = main_front(rs, latcol)
    front_labels = {ok[i]["label"] for i in front}
    yd = lambda r: min(r[latcol], clip_top) if clip_top else r[latcol]
    for i, r in enumerate(ok):
        fam, on = r["family"], i in front
        ax.scatter(r["size_kb"], yd(r), s=115 if on else 65, marker=MRK.get(fam, "o"), color=COL.get(fam, "#000"),
                   alpha=1 if on else 0.6, edgecolors="black" if on else "white", linewidths=1.4 if on else 0.6,
                   zorder=5 if on else 3, clip_on=False)
    fr = sorted(((ok[i]["size_kb"], ok[i][latcol]) for i in front))
    if len(fr) > 1:
        xs, ys = [p[0] for p in fr], [p[1] for p in fr]
        ax.step(xs + [ax.get_xlim()[1]], ys + [ys[-1]], where="post", color="#222", lw=1.3, alpha=0.7, zorder=2)
    # совпавшие точки подписываем одной подписью, остальные по отдельности
    buckets = {}
    for r in ok:
        buckets.setdefault((round(r["size_kb"] / 1.5), round(math.log(r[latcol]) / 0.015)), []).append(r)
    for b in buckets.values():
        x = sum(r["size_kb"] for r in b) / len(b)
        y = sum(yd(r) for r in b) / len(b)
        if len(b) > 1:
            ax.annotate(cluster_label(b), (x, y), textcoords="offset points", xytext=(-10, 18), ha="right",
                        va="bottom", fontsize=9, color="#444", arrowprops=dict(arrowstyle="-", color="#999", lw=0.7))
            continue
        r = b[0]
        on = r["label"] in front_labels
        name = {"int8_dyn": "INT8 dynamic", "int8_wo": "INT8 weight-only", "int8_full": "INT8 full-integer"}.get(
            r["precision"], r["variant"]) if r["precision"] != "fp32" else r["variant"]
        if r["precision"] != "fp32" and r["variant"].startswith("prune"):
            name = r["variant"] + " + " + name
        if (clip_top and r[latcol] > clip_top) or (not on and r["family"].startswith("int8")):
            ax.annotate(f"{name}\n{r[latcol]:.{0 if r[latcol] >= 100 else 1}f} мс" + ("  ↑ (вне шкалы)" if clip_top and r[latcol] > clip_top else ""), (x, y), textcoords="offset points",
                        xytext=(12, -6), ha="left", va="top", fontsize=8.5, color="#555")
            continue
        if on:
            off, ha, va = ((12, 0), "left", "center") if r["family"].startswith("int8") else ((0, -20), "center", "top")
        else:
            off, ha, va = (0, 10), "center", "bottom"
        ax.annotate(name, (x, y), textcoords="offset points", xytext=off, ha=ha, va=va,
                    fontsize=10.5 if on else 8.5, fontweight="bold" if on else "normal",
                    color="#111" if on else "#555")
    ax.set_xlabel("размер файла модели, КБ   (левее = лучше)")
    ax.set_ylabel("latency p50, мс   (ниже = лучше)")
    return front, ok, eq, lo, hi


def fig3_pareto(g, rs):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8))
    note = ""
    for ax, latcol, title in zip(axes, ("lat_b1_ms", "lat_b32_ms"), ("batch = 1", "batch = 32")):
        xs = [r["size_kb"] for r in rs]
        ax.set_xlim(min(xs) * 0.8, max(xs) * 1.08)
        vals = sorted(r[latcol] for r in rs)
        clip = None
        if vals[-1] > 1.5 * vals[len(vals) // 2]:  # один-два выброса сжимают остальное, режем шкалу
            rest = [v for v in vals if v <= 1.5 * vals[len(vals) // 2]]
            clip = max(rest) * 1.28
        top = clip if clip else vals[-1]
        span = top - vals[0]
        ax.set_ylim(vals[0] - span * 0.26, top + span * 0.12)
        front, ok, eq, lo, hi = draw_pareto(ax, rs, latcol, clip_top=clip)
        ax.set_title(title + ("   (шкала обрезана сверху)" if clip else ""), loc="left")
        note = (f"Точность в критерий не входит, 95% ДИ всех вариантов перекрываются ({lo * 100:.2f} ≤ {hi * 100:.2f}). "
                if eq else "Внимание, ДИ точности не перекрываются, точность включена в критерии. ")
    fams = [f for f in ORDER if any(r["family"] == f for r in rs)]
    handles = [Line2D([], [], marker=MRK[f], color=COL[f], ls="", ms=8, label=FAM_NAME[f]) for f in fams]
    handles.append(Line2D([], [], marker="o", color="w", markerfacecolor="#bbb", markeredgecolor="black", ms=9,
                          ls="", label="чёрный контур = на Парето-фронте"))
    fig.legend(handles=handles, loc="upper right", ncol=3, frameon=False, fontsize=9, bbox_to_anchor=(0.995, 0.995))
    fig.suptitle(f"Парето-фронт, размер × latency  ·  {g}", x=0.01, y=0.985, ha="left", fontsize=15, fontweight="bold")
    fig.text(0.01, 0.015, note + f"Допуск latency {args.lat_tol * 100:.0f}% (ничья). Ступенька = граница фронта.",
             fontsize=9.5, color="#555")
    fig.tight_layout(rect=(0, 0.04, 1, 0.9))
    fn = os.path.join(OUT, slug(g), "fig3_pareto.png")
    fig.savefig(fn, dpi=150)
    plt.close(fig)
    print("график:", fn)


def fig_all_runtimes():
    fig, ax = plt.subplots(figsize=(11, 6.2))
    rs = rows
    ok = [r for r in rs if r["lat_b1_ms"] and r["accuracy_b1"] is not None]
    objs = [Obj("size_kb", "min"), Obj("lat_b1_ms", "min", rel_tol=args.lat_tol)]
    front, _ = non_dominated(ok, objs)
    for i, r in enumerate(ok):
        on = i in front
        ax.scatter(r["lat_b1_ms"], r["size_kb"], s=100 if on else 55, marker=MRK.get(r["family"], "o"),
                   color=COL.get(r["family"], "#000"), alpha=1 if on else 0.4,
                   edgecolors="black" if on else "white", zorder=5 if on else 3)
        if on:
            ax.annotate(f"{r['label']}\n[{r['group']}]", (r["lat_b1_ms"], r["size_kb"]), textcoords="offset points",
                        xytext=(8, 6), fontsize=8.5)
    ax.set_xscale("log")
    ax.set_xlabel("latency batch=1, мс p50 (log)")
    ax.set_ylabel("размер файла, КБ")
    ax.set_title("Все рантаймы вместе, размер × latency (b=1). Контур это Парето-фронт", loc="left")
    fig.tight_layout()
    fn = os.path.join(OUT, "all_runtimes_lat_b1.png")
    fig.savefig(fn, dpi=150)
    plt.close(fig)
    print("график:", fn)


make_table()
for g in groups:
    rs = [r for r in rows if r["group"] == g]
    os.makedirs(os.path.join(OUT, slug(g)), exist_ok=True)
    report_group(g, rs)
    fig1_tradeoff(g, rs)
    fig2_accuracy(g, rs)
    fig3_pareto(g, rs)
if len(runtimes) > 1:
    fig_all_runtimes()

with open(os.path.join(OUT, "pareto_report.md"), "w") as fh:
    fh.write("\n".join(report) + "\n")
with open(os.path.join(OUT, "pareto_membership.csv"), "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["group", "plane", "label", "on_strict_front", "on_robust_front", "dominated_by_strict"])
    w.writerows(membership)
print(f"\nотчёт: {os.path.join(OUT, 'pareto_report.md')}\nтаблица: {os.path.join(OUT, 'table1.md')}")
