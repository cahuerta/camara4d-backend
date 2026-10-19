"""toma.py — Leer y escribir tomas (formato «camara4d-toma» v1, el mismo que genera el teléfono).

Los arreglos binarios viajan en base64 (Float32 / Uint8 / Uint32, little-endian).
"""
import base64
import re

import numpy as np

_ID = re.compile(r"^[\w.-]{1,80}$")


def dec(b64, dtype):
    """base64 → arreglo numpy (copia propia, alineada)."""
    return np.frombuffer(base64.b64decode(b64), dtype=np.dtype(dtype).newbyteorder("<")).astype(dtype)


def enc(arr, dtype):
    """arreglo numpy → base64."""
    return base64.b64encode(np.ascontiguousarray(arr, dtype=np.dtype(dtype).newbyteorder("<")).tobytes()).decode("ascii")


def validate(T):
    if not isinstance(T, dict) or T.get("formato") != "camara4d-toma":
        raise ValueError("no es una toma de Cámara 4D")
    if not isinstance(T.get("id"), str) or not _ID.match(T["id"]):
        raise ValueError("id de toma inválido")
    p = T.get("puntos")
    if not isinstance(p, dict) or not isinstance(p.get("n"), int) or not p.get("xyz"):
        raise ValueError("la toma no trae puntos")
    if not isinstance(T.get("vistas"), list):
        raise ValueError("la toma no trae vistas")


class Vista:
    """Vista guardada: pose (p_cam = R·X + t), firmas de puntos y su posición 3D en la toma."""
    __slots__ = ("R", "t", "n", "kp", "desc", "X")

    def __init__(self, R, t, n, kp, desc, X):
        self.R, self.t, self.n, self.kp, self.desc, self.X = R, t, n, kp, desc, X


class Trabajo:
    """Toma decodificada para trabajar con numpy."""

    def __init__(self, meta, voxel, obj, xyz, rgb, views, kfs):
        self.meta, self.voxel, self.obj = meta, voxel, obj
        self.xyz = xyz        # (n, 3) float64
        self.rgb = rgb        # (n, 3) uint8
        self.views = views    # (n,)   uint8
        self.kfs = kfs        # [Vista]

    @property
    def n(self):
        return len(self.xyz)


def decode(T):
    n = T["puntos"]["n"]
    xyz = dec(T["puntos"]["xyz"], np.float32).reshape(n, 3).astype(np.float64)
    rgb = dec(T["puntos"]["rgb"], np.uint8).reshape(n, 3)
    views = dec(T["puntos"]["vistas"], np.uint8)
    kfs = []
    for v in T["vistas"]:
        k = int(v["n"])
        kfs.append(Vista(
            R=np.array(v["R"], dtype=np.float64).reshape(3, 3),
            t=np.array(v["t"], dtype=np.float64),
            n=k,
            kp=dec(v["kp"], np.float32),
            desc=dec(v["desc"], np.uint32).reshape(k, 8),
            X=dec(v["X"], np.float32).reshape(k, 3).astype(np.float64),
        ))
    obj = T.get("objeto")
    if obj:
        obj = {"c": np.array(obj["c"], dtype=np.float64), "r": float(obj["r"])}
    return Trabajo(T, float(T.get("voxel") or 0.006), obj, xyz, rgb, views, kfs)


def summary(T):
    return {"id": T["id"], "nombre": T.get("nombre"), "fecha": T.get("fecha"), "origen": T.get("origen"),
            "puntos": T["puntos"]["n"], "vistas": len(T["vistas"])}
  
