"""sesion.py — SESIONES CON VARIAS CÁMARAS: el servidor es el centro.

Una sesión es "lo que queremos reconocer" (por ejemplo «rodilla» o «taza»). Se crea con un nombre y entrega un
código corto; cada teléfono que se une con ese código pasa a ser una cámara de la sesión. Todas usan la misma
hoja como referencia, así que sus vistas ya están en el mismo espacio (no hace falta sincronizar relojes).

Cada vista que llega trae: posición de la cámara respecto de la hoja, la imagen (JPEG), la máscara de dónde se
ve la hoja (0 sin información, 1 se ve la hoja, 2 algo la tapa) y, más tarde, su mapa de profundidad.
El servidor arma un solo modelo con las vistas de todas las cámaras, por capas:
  1. FORMA: un bloque sobre la zona libre de la hoja que cada vista talla (lo que deja ver la hoja está vacío).
  2. RELIEVE: los mapas de profundidad vacían lo que está claramente delante de la superficie medida.
  3. COLOR: cada cubito de la piel toma el color de las vistas que lo ven sin nada delante.
Todo con numpy (cálculo propio; scipy solo para operaciones de grilla: vecindades y grupos conectados).
"""
import base64
import io
import random
import string
import threading
import time

import numpy as np
from PIL import Image
from scipy import ndimage

# Zona libre de la hoja (metros) — debe coincidir con js/hoja.js (ZONA)
ZONA = (-0.05995, 0.05995, -0.0917, 0.0917)
VOX = 0.0015          # cubito de 1,5 mm (más fino que en el teléfono)
HMAX = 0.16           # altura máxima del objeto

sesiones = {}
_lock = threading.Lock()


def _codigo():
    while True:
        c = "".join(random.choices(string.ascii_uppercase.replace("O", "").replace("I", ""), k=4))
        if c not in sesiones:
            return c


def _unpack_mask(b64, w, h):
    raw = np.frombuffer(base64.b64decode(b64), np.uint8)
    m = np.empty(raw.size * 4, np.uint8)
    for k in range(4):
        m[k::4] = (raw >> (2 * k)) & 3
    return m[: w * h].reshape(h, w)


class Volumen:
    def __init__(self):
        x0, x1, y0, y1 = ZONA
        self.nx, self.ny, self.nz = int(np.ceil((x1 - x0) / VOX)), int(np.ceil((y1 - y0) / VOX)), int(np.ceil(HMAX / VOX))
        n = self.nx * self.ny * self.nz
        iz, iy, ix = np.meshgrid(np.arange(self.nz), np.arange(self.ny), np.arange(self.nx), indexing="ij")
        self.C = np.stack([x0 + (ix.ravel() + 0.5) * VOX, y0 + (iy.ravel() + 0.5) * VOX, (iz.ravel() + 0.5) * VOX], axis=1).astype(np.float32)
        self.empty = np.zeros(n, np.uint16)
        self.seen = np.zeros(n, np.uint16)
        self.free = np.zeros(n, np.uint16)
        self.vistas = 0

    def _proyectar(self, R, t, f, cx, cy, w, h, sel=None):
        P = self.C if sel is None else self.C[sel]
        R = np.asarray(R, np.float32).reshape(3, 3)
        q = P @ R.T + np.asarray(t, np.float32)
        z = q[:, 2]
        ok = z > 0.01
        zs = np.where(ok, z, 1)
        u = np.rint(f * q[:, 0] / zs + cx).astype(np.int32)
        v = np.rint(f * q[:, 1] / zs + cy).astype(np.int32)
        ok &= (u >= 0) & (v >= 0) & (u < w) & (v < h)
        return u, v, z, ok

    # 1. FORMA
    def tallar(self, mask, R, t, f, cx, cy):
        h, w = mask.shape
        u, v, _, ok = self._proyectar(R, t, f, cx, cy, w, h)
        m = np.zeros(len(u), np.uint8)
        m[ok] = mask[v[ok], u[ok]]
        self.empty += (m == 1)
        self.seen += (m == 2)
        self.vistas += 1

    # 2. RELIEVE
    def relieve(self, depth, R, t, f, cx, cy):
        h, w = depth.shape
        dmin = ndimage.minimum_filter(depth, size=3, mode="constant", cval=0)   # 0 = sin medida → no vacía
        sel = np.nonzero(self.seen > 0)[0]
        u, v, z, ok = self._proyectar(R, t, f, cx, cy, w, h, sel)
        d = np.zeros(len(sel), np.float32)
        d[ok] = dmin[v[ok], u[ok]]
        hit = ok & (d > z + np.maximum(3 * VOX, 0.02 * z))
        self.free[sel[hit]] += 1

    def lleno(self):
        e, s, fr = self.empty.astype(np.int32), self.seen.astype(np.int32), self.free.astype(np.int32)
        F = (s >= 3) & (e <= np.maximum(1, 0.1 * (e + s))) & ~((fr >= 2) & (fr * 3 > s))
        return F.reshape(self.nz, self.ny, self.nx)

    # La piel: cubitos llenos que tocan el vacío de AFUERA (los huecos cerrados por dentro se rellenan)
    def piel(self):
        F = self.lleno()
        vacio = ~F
        lab, n = ndimage.label(vacio)
        borde = np.zeros(n + 1, bool)
        for cara in (lab[:, :, 0], lab[:, :, -1], lab[:, 0, :], lab[:, -1, :], lab[-1, :, :]):
            borde[np.unique(cara)] = True
        borde[0] = False
        afuera = borde[lab]
        F = ~afuera
        toca = ndimage.binary_dilation(afuera) & F
        toca[0] = False   # el piso es la hoja
        return F, toca


class Sesion:
    def __init__(self, nombre):
        self.codigo = _codigo()
        self.nombre = nombre or "objeto"
        self.creada = time.time()
        self.camaras = {}      # id → {nombre, vistas, ultima}
        self.vistas = {}       # id → {R, t, f, cx, cy, img (media resolución), camara}
        self.vol = Volumen()
        self.version = 0
        self._cache = None     # (versión, modelo)
        self.lock = threading.Lock()
        self.mlock = threading.Lock()   # un cálculo del modelo a la vez (los demás reciben el mismo resultado)

    def resumen(self):
        return {"codigo": self.codigo, "nombre": self.nombre, "creada": self.creada, "version": self.version,
                "vistas": len(self.vistas), "vistas_talladas": self.vol.vistas,
                "camaras": [{"id": k, **v} for k, v in self.camaras.items()]}

    def unir_camara(self, nombre):
        cid = "c" + "".join(random.choices(string.ascii_lowercase + string.digits, k=5))
        self.camaras[cid] = {"nombre": nombre or f"Cámara {len(self.camaras) + 1}", "vistas": 0, "ultima": time.time()}
        return cid

    def agregar_vista(self, d):
        cam = d.get("camara")
        if cam not in self.camaras:
            raise ValueError("cámara no registrada en la sesión")
        w, h = int(d["w"]), int(d["h"])
        mask = _unpack_mask(d["mask"], w, h)
        img = Image.open(io.BytesIO(base64.b64decode(d["jpg"]))).convert("RGB")
        half = np.asarray(img.resize((w // 2, h // 2), Image.BILINEAR), np.uint8)
        vid = f'{cam}-{d["id"]}'
        with self.lock:
            self.vol.tallar(mask, d["R"], d["t"], float(d["f"]), float(d["cx"]), float(d["cy"]))
            self.vistas[vid] = {"R": d["R"], "t": d["t"], "f": float(d["f"]), "cx": float(d["cx"]), "cy": float(d["cy"]), "img": half, "camara": cam}
            self.camaras[cam]["vistas"] += 1
            self.camaras[cam]["ultima"] = time.time()
            self.version += 1
        return vid

    def agregar_profundidad(self, vid, d):
        V = self.vistas.get(vid)
        if not V:
            raise ValueError("vista no encontrada")
        w, h = int(d["w"]), int(d["h"])
        dep = np.frombuffer(base64.b64decode(d["mm10"]), np.uint16)[: w * h].reshape(h, w).astype(np.float32) / 10000.0
        with self.lock:
            self.vol.relieve(dep, V["R"], V["t"], float(d["f"]), float(d["cx"]), float(d["cy"]))
            self.version += 1

    # 3. COLOR y resultado
    def modelo(self):
        with self.mlock:
            return self._modelo()

    def _modelo(self):
        # Se copia el estado (rápido) y se calcula sin bloquear: las vistas siguen llegando mientras tanto
        with self.lock:
            if self._cache and self._cache[0] == self.version:
                return self._cache[1]
            version = self.version
            vol = Volumen.__new__(Volumen)
            vol.__dict__.update(self.vol.__dict__)
            vol.empty, vol.seen, vol.free = self.vol.empty.copy(), self.vol.seen.copy(), self.vol.free.copy()
            vistas = list(self.vistas.values())
        if True:
            F, piel = vol.piel()
            idx = np.flatnonzero(piel.ravel())
            if len(idx) == 0:
                res = {"version": version, "n": 0}
                self._cache = (version, res)
                return res
            P = vol.C[idx]
            col = np.zeros((len(idx), 3), np.float64)
            wsum = np.zeros(len(idx), np.float64)
            l1 = np.zeros(len(idx)); l2 = np.zeros(len(idx)); ln = np.zeros(len(idx))
            for V in vistas:
                img = V["img"]
                h, w = img.shape[:2]
                f, cx, cy = V["f"] / 2, V["cx"] / 2, V["cy"] / 2
                R = np.asarray(V["R"], np.float64).reshape(3, 3)
                q = P @ R.T + np.asarray(V["t"])
                z = q[:, 2]
                ok = z > 0.01
                zs = np.where(ok, z, 1)
                uf, vf = f * q[:, 0] / zs + cx, f * q[:, 1] / zs + cy
                u, v = np.floor(uf).astype(np.int64), np.floor(vf).astype(np.int64)
                ok &= (u >= 0) & (v >= 0) & (u < w) & (v < h)
                # Lo más cercano en cada píxel (con el tamaño del cubito en pantalla): lo de atrás queda tapado
                zb = np.full(w * h, np.inf)
                rad = np.clip(np.rint(0.6 * f * VOX / zs), 1, 4).astype(np.int64)
                for dv in range(-2, 3):
                    for du in range(-2, 3):
                        sel = ok & (np.abs(du) <= rad) & (np.abs(dv) <= rad)
                        uu, vv = u[sel] + du, v[sel] + dv
                        k = (uu >= 0) & (vv >= 0) & (uu < w) & (vv < h)
                        np.minimum.at(zb, vv[k] * w + uu[k], z[sel][k])
                vis = ok.copy()
                vis[ok] = zb[v[ok] * w + u[ok]] >= z[ok] - 2.5 * VOX
                c = img[v[vis], u[vis]].astype(np.float64)
                wt = 1.0 / z[vis] ** 2
                col[vis] += c * wt[:, None]
                wsum[vis] += wt
                L = c @ np.array([0.299, 0.587, 0.114])
                l1[vis] += L; l2[vis] += L * L; ln[vis] += 1
            rgb = np.where(wsum[:, None] > 0, col / np.maximum(wsum, 1e-12)[:, None], np.nan)
            # Un cubito real se ve del mismo color desde todas las vistas: si no, su color se toma de los vecinos
            sd = np.sqrt(np.maximum(0, l2 / np.maximum(ln, 1) - (l1 / np.maximum(ln, 1)) ** 2))
            rgb[(ln >= 3) & (sd > 38)] = np.nan
            malo = np.isnan(rgb[:, 0])
            if malo.any() and (~malo).any():
                grid = np.full(vol.nz * vol.ny * vol.nx, -1, np.int64)
                grid[idx[~malo]] = np.arange(len(idx))[~malo]
                hueco = grid.reshape(vol.nz, vol.ny, vol.nx) < 0
                _, ind = ndimage.distance_transform_edt(hueco, return_indices=True)
                near = grid.reshape(vol.nz, vol.ny, vol.nx)[ind[0], ind[1], ind[2]].ravel()[idx[malo]]
                rgb[malo] = rgb[near]
            rgb = np.nan_to_num(rgb, nan=150)
            # Motas sueltas: se deja el bloque mayor (y los de al menos 1/6 de su tamaño)
            lab, n = ndimage.label(piel, structure=np.ones((3, 3, 3)))
            if n > 1:
                sizes = np.bincount(lab.ravel())[1:]
                keep_l = np.flatnonzero(sizes >= sizes.max() / 6) + 1
                keep = np.isin(lab.ravel()[idx], keep_l)
                P, rgb = P[keep], rgb[keep]
            size_mm = [int(round(1000 * (P[:, k].max() - P[:, k].min()))) for k in range(3)]
            res = {
                "version": version, "n": int(len(P)), "voxel": VOX, "vistas": len(vistas),
                "medidas_mm": size_mm,
                "xyz": base64.b64encode(P.astype("<f4").tobytes()).decode(),
                "rgb": base64.b64encode(np.clip(np.rint(rgb), 0, 255).astype(np.uint8).tobytes()).decode(),
            }
            self._cache = (version, res)
            return res


def crear(nombre):
    with _lock:
        # Se olvidan las sesiones de más de 12 horas
        for k in [k for k, s in sesiones.items() if time.time() - s.creada > 12 * 3600]:
            sesiones.pop(k, None)
        s = Sesion(nombre)
        sesiones[s.codigo] = s
        return s


def buscar(codigo):
    s = sesiones.get((codigo or "").upper())
    if not s:
        raise KeyError("sesión no encontrada (o el servidor se reinició)")
    return s
