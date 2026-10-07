# точное множество недоминируемых точек по любому набору критериев.
#
# a доминирует b, если a не хуже b по всем критериям и строго лучше хотя бы по одному.
# в строгом режиме сравниваются точные числа. в робастном разница меньше шума считается ничьёй,
# а шум берём из данных. для accuracy это абсолютный допуск (0.1 п.п., примерно одна сигма
# биномиальной ошибки на 10k), для latency большее из относительного допуска и разброса между
# раундами, размер детерминирован, так что для него допуск 0
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Obj:
    col: str
    sense: str                       # "min" или "max"
    abs_tol: float = 0.0
    rel_tol: float = 0.0
    spread_col: Optional[str] = None  # колонка с относительным разбросом измерения (0.03 = 3%)


def _tol(o, ra, rb):
    a, b = ra[o.col], rb[o.col]
    base = max(abs(a), abs(b))
    t = o.abs_tol + o.rel_tol * base
    if o.spread_col:
        sa = ra.get(o.spread_col) or 0.0
        sb = rb.get(o.spread_col) or 0.0
        t = max(t, max(sa, sb) * base)
    return t


def _cmp(o, ra, rb):
    # 1 если ra лучше, -1 если хуже, 0 если ничья в пределах допуска
    a, b = ra[o.col], rb[o.col]
    d = (b - a) if o.sense == "min" else (a - b)
    t = _tol(o, ra, rb)
    if d > t:
        return 1
    if d < -t:
        return -1
    return 0


def dominates(ra, rb, objs: List[Obj]):
    c = [_cmp(o, ra, rb) for o in objs]
    return all(v >= 0 for v in c) and any(v > 0 for v in c)


def non_dominated(rows, objs: List[Obj]):
    # возвращает индексы фронта и словарь, где у каждой доминируемой точки список тех, кто её доминирует
    front, dom_by = [], {}
    for i, r in enumerate(rows):
        doms = [j for j, q in enumerate(rows) if j != i and dominates(q, r, objs)]
        if doms:
            dom_by[i] = doms
        else:
            front.append(i)
    return front, dom_by


def objectives(plane, robust=False, acc_tol=0.001, lat_tol_rel=0.03):
    # plane бывает size, lat_b1, lat_b32 и size_lat_b1 (3D).
    # accuracy берём в том же режиме, что и вторая ось, то есть batch=1 для lat_b1 и
    # batch=32 для size и lat_b32. для динамического int8 это важно, у него точность зависит от батча
    def acc(col):
        return Obj(col, "max", abs_tol=acc_tol if robust else 0.0)

    size = Obj("size_kb", "min")

    def lat(col):
        return Obj(col, "min", rel_tol=lat_tol_rel if robust else 0.0,
                   spread_col=(col + "_spread") if robust else None)
    return {
        "size": [size, acc("accuracy_b32")],
        "lat_b1": [lat("lat_b1_ms"), acc("accuracy_b1")],
        "lat_b32": [lat("lat_b32_ms"), acc("accuracy_b32")],
        "size_lat_b1": [size, lat("lat_b1_ms"), acc("accuracy_b1")],
    }[plane]


def family(variant, precision):
    # к какому семейству метода относится точка (для легенды и выводов)
    v, p = variant.lower(), precision.lower()
    if p.startswith("int8"):
        if v.startswith("prune"):
            return "structured+int8"
        return "int8-" + p.replace("int8_", "")      # int8-dyn / int8-wo / int8-full
    if v.startswith("unstruct"):
        return "unstructured"
    if v.startswith("prune"):
        return "structured"
    return "baseline"


def analyse(rows, plane, robust=False, **kw):
    objs = objectives(plane, robust=robust, **kw)
    ok = [r for r in rows if all(isinstance(r.get(o.col), (int, float)) for o in objs)]
    front, dom_by = non_dominated(ok, objs)
    return ok, front, dom_by, objs
