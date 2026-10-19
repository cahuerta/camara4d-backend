"""merge.py — Une tomas del mismo objeto (de distintos momentos, teléfonos o cámaras) en un solo modelo 3D.

Cada toma tiene su propio espacio, escala y punto de partida. Para unirlas:
 1. Se buscan zonas en común: firmas de puntos (256 bits) que aparecen en ambas tomas → pares de puntos 3D.
 2. Muestreo aleatorio (RANSAC) de 3 pares → transformación de semejanza (giro + traslado + escala, método de
    Umeyama); gana la que explica más pares. Se prueban miles de candidatas a la vez (numpy).
 3. Ajuste fino con las superficies completas (ICP): se acerca cada punto de una toma al más cercano de la otra.
 4. Fusión: todos los puntos en un solo espacio de cubitos (vóxeles), promediando posición y color.
Si una toma no comparte ninguna zona con las demás, se informa y queda fuera.

Todo el cálculo es propio; numpy solo hace las cuentas con arreglos y scipy aporta el buscador de vecinos (árbol k-d).
"""
import random
import string
import time
from datetime import datetime, timezone

import numpy as np
from scipy.spatial import cKDTree

from toma import Trabajo, Vista, decode, enc

RNG = np.random.default_rng()


# ---------- Semejanza (Umeyama), para muchas a la vez ----------
def umeyama_batch(A, B):
    """A, B: (k, m, 3). Devuelve s (k,), R (k,3,3), t (k,3) que minimizan Σ |s·R·b + t − a|², y una máscara de válidas."""
    ma, mb = A.mean(axis=1), B.mean(axis=1)
    a, b = A - ma[:, None, :], B - mb[:, None, :]
    m = A.shape[1]
    vb = (b * b).sum(axis=(1, 2)) / m
    Sg = np.einsum("kmi,kmj->kij", a, b) / m
    U, S, Vt = np.linalg.svd(Sg)
    d = np.where(np.linalg.det(U) * np.linalg.det(Vt) < 0, -1.0, 1.0)
    D = np.ones((len(A), 3))
    D[:, 2] = d
    R = U @ (D[:, :, None] * Vt)
    ok = vb > 1e-12
    s = np.where(ok, (S * D).sum(axis=1) / np.where(ok, vb, 1), 0)
    t = ma - s[:, None] * np.einsum("kij,kj->ki", R, mb)
    return s, R, t, ok


def umeyama(A, B):
    s, R, t, ok = umeyama_batch(A[None], B[None])
    return (float(s[0]), R[0], t[0]) if ok[0] else None


def apply(T, P):
    s, R, t = T
    return s * (P @ R.T) + t


# ---------- Firmas ----------
def pool_features(W: Trabajo, max_n=7000):
    """Junta las firmas de todas las vistas de una toma (con su punto 3D), con un máximo para que sea rápido."""
    if not W.kfs:
        return np.zeros((0, 4), np.uint64), np.zeros((0, 3))
    desc = np.concatenate([k.desc for k in W.kfs])
    X = np.concatenate([k.X for k in W.kfs])
    good = np.isfinite(X).all(axis=1)
    desc, X = desc[good], X[good]
    if len(desc) > max_n:
        idx = np.linspace(0, len(desc) - 1, max_n).astype(int)
        desc, X = desc[idx], X[idx]
    return np.ascontiguousarray(desc).view(np.uint64), X   # 8×32 bits → 4×64 bits


def hamming_rows(qb, qa):
    """Distancias de Hamming (filas de qb contra todas las de qa) → (len(qb), len(qa))."""
    D = np.bitwise_count(qb[:, None, 0] ^ qa[None, :, 0]).astype(np.uint16)
    for k in range(1, qa.shape[1]):
        D += np.bitwise_count(qb[:, None, k] ^ qa[None, :, k])
    return D


def match_pools(da, Xa, db, Xb, sep, max_dist=64, ratio=0.8, avance=None):
    """Pareja para cada firma de B: la más parecida de A, si es claramente mejor que la segunda.
    El mismo punto aparece en varias vistas de A, así que la segunda se busca solo entre puntos de otro lugar
    (a más de sep del primero); si no, el mismo punto repetido haría fallar la comparación."""
    ia, ib, dd = [], [], []
    sep2 = sep * sep
    na = (Xa * Xa).sum(axis=1)
    for c0 in range(0, len(db), 512):
        if avance:
            avance(c0 / max(1, len(db)))
        D = hamming_rows(db[c0:c0 + 512], da)
        j1 = D.argmin(axis=1)
        b1 = D[np.arange(len(D)), j1].astype(np.int32)
        cand = np.nonzero(b1 <= max_dist)[0]
        if not len(cand):
            continue
        Y = Xa[j1[cand]]
        # Distancia² de cada punto de A al de la mejor pareja (con producto de matrices: rápido)
        d2 = na[None, :] - 2 * (Y @ Xa.T) + (Y * Y).sum(axis=1)[:, None]
        Dc = D[cand].astype(np.int32)
        Dc[d2 <= sep2] = 999
        b2 = Dc.min(axis=1)
        ok = b1[cand] < ratio * b2
        sel = cand[ok]
        ia.append(j1[sel]); ib.append(sel + c0); dd.append(b1[sel])
    ia, ib, dd = np.concatenate(ia), np.concatenate(ib), np.concatenate(dd)
    o = np.argsort(dd, kind="stable")
    return Xa[ia[o]], Xb[ib[o]]


# ---------- Alinear la toma B con el modelo A ----------
def points_of(W: Trabajo, min_views, max_n):
    P = W.xyz[W.views >= min_views]
    if len(P) > max_n:
        P = P[np.linspace(0, len(P) - 1, max_n).astype(int)]
    return P


def _nada(frac, etapa=None):
    pass


def align(A: Trabajo, B: Trabajo, avance=_nada):
    """avance(fracción 0–1, etapa): informa cuánto lleva (para la barra de la app)."""
    da, Xa = pool_features(A)
    db, Xb = pool_features(B)
    if len(da) == 0 or len(db) == 0:
        return {"ok": False, "motivo": "una de las tomas no trae firmas de puntos"}
    tau = 3 * A.voxel
    avance(0.0, "buscando zonas en común")
    PA_m, PB_m = match_pools(da, Xa, db, Xb, 4 * tau, avance=lambda f: avance(0.45 * f, "buscando zonas en común"))
    n = len(PA_m)
    if n < 12:
        return {"ok": False, "motivo": f"no comparten zonas reconocibles ({n} coincidencias)"}

    # RANSAC de semejanza: primero se prueban las parejas más parecidas (el grupo se amplía de a poco)
    best, best_inl = None, np.zeros(0, int)
    ITER, LOTE = 40000, 4000
    for it0 in range(0, ITER, LOTE):
        avance(0.45 + 0.35 * it0 / ITER, "alineando")
        it = np.arange(it0, it0 + LOTE)
        pool = np.minimum(n, 30 + it // 40)
        I = (RNG.random((LOTE, 3)) * pool[:, None]).astype(int)
        I = I[(I[:, 0] != I[:, 1]) & (I[:, 1] != I[:, 2]) & (I[:, 0] != I[:, 2])]
        SA, SB = PA_m[I], PB_m[I]                                   # (k, 3, 3)
        da_ = np.linalg.norm(SA[:, [0, 0, 1]] - SA[:, [1, 2, 2]], axis=2)
        db_ = np.linalg.norm(SB[:, [0, 0, 1]] - SB[:, [1, 2, 2]], axis=2)
        # Tríos no muy juntos y con la misma forma en A y en B (otra escala)
        keep = (da_ >= 3 * tau).all(axis=1) & (db_ > 0).all(axis=1)
        r = da_ / np.where(db_ > 0, db_, 1)
        keep &= r.max(axis=1) < 1.15 * r.min(axis=1)
        if not keep.any():
            continue
        s, R, t, ok = umeyama_batch(SA[keep], SB[keep])
        ok &= (s > 0.05) & (s < 20)
        s, R, t = s[ok], R[ok], t[ok]
        for c0 in range(0, len(s), 500):
            Q = s[c0:c0 + 500, None, None] * np.einsum("kij,pj->kpi", R[c0:c0 + 500], PB_m) + t[c0:c0 + 500, None, :]
            inl = ((Q - PA_m[None]) ** 2).sum(axis=2) < tau * tau
            cnt = inl.sum(axis=1)
            k = int(cnt.argmax())
            if cnt[k] > len(best_inl):
                best = (float(s[c0 + k]), R[c0 + k], t[c0 + k])
                best_inl = np.nonzero(inl[k])[0]
    if best is None or len(best_inl) < 20 or len(best_inl) < 0.06 * n:
        return {"ok": False, "motivo": f"las coincidencias no forman una zona común consistente ({len(best_inl)} de {n})"}
    T = umeyama(PA_m[best_inl], PB_m[best_inl]) or best

    # Ajuste fino con las superficies (ICP con escala)
    PA, PB = points_of(A, 2, 200000), points_of(B, 2, 50000)
    tree = cKDTree(PA)

    def mean_dist(T0):
        d, _ = tree.query(apply(T0, PB), distance_upper_bound=1.5 * tau)
        f = np.isfinite(d)
        return (float(d[f].mean()) if f.any() else np.inf), int(f.sum())

    cur_d, cur_n = mean_dist(T)
    for it in range(20):
        avance(0.8 + 0.18 * it / 20, "ajuste fino de superficies")
        thr = 3 * tau if it < 5 else 1.5 * tau
        d, j = tree.query(apply(T, PB), distance_upper_bound=thr)
        f = np.isfinite(d)
        if f.sum() < 50:
            break
        Tn = umeyama(PA[j[f]], PB[f])
        if Tn is None or not (0.5 * T[0] < Tn[0] < 2 * T[0]):
            break
        nd, nn = mean_dist(Tn)
        if nn < 0.8 * cur_n or nd > cur_d * 1.001:
            break
        T, cur_d, cur_n = Tn, nd, nn
    return {"ok": True, "T": T, "coincidencias": int(len(best_inl)), "pares": n,
            "solape": cur_n / max(1, len(PB)), "error": cur_d}


# ---------- Fusión ----------
def fuse(A: Trabajo, B: Trabajo, T):
    s = A.voxel
    P = np.concatenate([A.xyz, apply(T, B.xyz)])
    C = np.concatenate([A.rgb, B.rgb]).astype(np.float64)
    w = np.maximum(1, np.concatenate([A.views, B.views])).astype(np.float64)
    I = np.floor(P / s).astype(np.int64) + (1 << 20)
    key = (I[:, 0] << 42) | (I[:, 1] << 21) | I[:, 2]
    _, inv = np.unique(key, return_inverse=True)
    m = inv.max() + 1
    W = np.bincount(inv, weights=w, minlength=m)
    xyz = np.stack([np.bincount(inv, weights=P[:, k] * w, minlength=m) for k in range(3)], axis=1) / W[:, None]
    rgb = np.stack([np.bincount(inv, weights=C[:, k] * w, minlength=m) for k in range(3)], axis=1) / W[:, None]
    views = np.minimum(255, W).astype(np.uint8)

    # Vistas de B llevadas al espacio de A (sirven para unir una tercera toma)
    ts, TR, Tt = T
    kfs = list(A.kfs)
    for k in B.kfs:
        Rp = k.R @ TR.T
        tp = ts * k.t - Rp @ Tt
        kfs.append(Vista(Rp, tp, k.n, k.kp, k.desc, apply(T, k.X)))

    # Esfera del objeto que contiene a ambas
    obj = A.obj
    if A.obj and B.obj:
        cb, rb = apply(T, B.obj["c"][None])[0], B.obj["r"] * ts
        c = (A.obj["c"] + cb) / 2
        obj = {"c": c, "r": float(max(np.linalg.norm(c - A.obj["c"]) + A.obj["r"], np.linalg.norm(c - cb) + rb))}
    return Trabajo(A.meta, A.voxel, obj, xyz, np.clip(np.rint(rgb), 0, 255).astype(np.uint8), views, kfs)


# ---------- Unir varias tomas ----------
def _miles(n):
    return f"{n:,}".replace(",", ".")


def merge_tomas(tomas, avance=_nada):
    """avance(fracción 0–1, etapa) se llama durante el trabajo (para mostrar una barra de avance)."""
    if len(tomas) < 2:
        raise ValueError("se necesitan al menos 2 tomas")
    avance(0.0, "leyendo tomas")
    works = [decode(T_) for T_ in tomas]
    # Base: la toma con más puntos. Las demás se prueban contra lo ya unido, en varias pasadas (una que no calzaba
    # al principio puede calzar después, cuando lo unido ya cubre más caras)
    orden = sorted(range(len(tomas)), key=lambda i: -works[i].n)
    base = orden[0]
    acc = works[base]
    detalle = {base: {"id": tomas[base]["id"], "nombre": tomas[base].get("nombre"), "unida": True, "base": True}}
    pendientes = orden[1:]
    motivos = {}
    total = max(1, len(pendientes))
    hechos = 0
    cambio = True
    while pendientes and cambio:
        cambio = False
        for i in list(pendientes):
            ini = min(hechos, total - 1) / total
            nombre = tomas[i].get("nombre") or tomas[i]["id"]
            r = align(acc, works[i], avance=lambda f, e=None, ini=ini, nombre=nombre: avance(0.02 + 0.93 * (ini + f / total), f"{e or 'alineando'} · {nombre}"))
            if not r["ok"]:
                motivos[i] = r["motivo"]
                continue
            avance(0.02 + 0.93 * (ini + 0.99 / total), f"fusionando · {nombre}")
            acc = fuse(acc, works[i], r["T"])
            pendientes.remove(i)
            hechos += 1
            cambio = True
            detalle[i] = {
                "id": tomas[i]["id"], "nombre": tomas[i].get("nombre"), "unida": True, "coincidencias": r["coincidencias"],
                "solape": round(100 * r["solape"]), "error": round(100 * r["error"] / (acc.obj["r"] if acc.obj else 1), 2),
                "escala": round(r["T"][0], 4),
            }
    for i in pendientes:
        detalle[i] = {"id": tomas[i]["id"], "nombre": tomas[i].get("nombre"), "unida": False, "motivo": motivos.get(i, "sin zona en común")}
    detalle = [detalle[i] for i in range(len(tomas))]
    avance(0.96, "guardando el resultado")
    unidas = sum(1 for d in detalle if d["unida"])
    ahora = datetime.now(timezone.utc)
    sufijo = "".join(random.choices(string.ascii_lowercase + string.digits, k=4))
    toma = {
        "formato": "camara4d-toma", "version": 1,
        "id": f"union-{np.base_repr(int(time.time() * 1000), 36).lower()}-{sufijo}",
        "nombre": f"Unión de {unidas} tomas", "fecha": ahora.isoformat(timespec="milliseconds").replace("+00:00", "Z"),
        "origen": "servidor",
        "voxel": acc.voxel, "intr": tomas[0].get("intr"), "hoja": tomas[base].get("hoja"),
        "objeto": {"c": acc.obj["c"].tolist(), "r": acc.obj["r"]} if acc.obj else None,
        "mapas": 3,
        "partes": [d["id"] for d in detalle if d["unida"]],
        "puntos": {"n": int(acc.n), "xyz": enc(acc.xyz, np.float32), "rgb": enc(acc.rgb, np.uint8), "vistas": enc(acc.views, np.uint8)},
        "vistas": [{"R": k.R.reshape(-1).tolist(), "t": k.t.tolist(), "n": k.n, "kp": enc(k.kp, np.float32),
                    "desc": enc(k.desc, np.uint32), "X": enc(k.X, np.float32)} for k in acc.kfs],
    }
    fallidas = [d for d in detalle if not d["unida"]]
    informe = f"{unidas} de {len(tomas)} tomas unidas en un modelo de {_miles(acc.n)} puntos."
    if fallidas:
                informe += (f" Quedaron fuera {len(fallidas)} por no tener una zona en común clara con el resto: "
                    + ", ".join(f"«{d['nombre'] or d['id']}»" for d in fallidas) + ".")
    return {"toma": toma, "informe": informe, "detalle": detalle}
