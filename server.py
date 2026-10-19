"""server.py — Servidor de Cámara 4D (Python: FastAPI + numpy).

 · Recibe las tomas de los teléfonos y las guarda.
 · Une tomas del mismo objeto (hechas en momentos distintos, con otros teléfonos o cámaras) en un solo modelo.

Rutas (las mismas que usa la app):
  GET    /health            estado del servidor
  GET    /tomas             lista de tomas (resumen)
  HEAD   /tomas/{id}        ¿existe la toma?
  GET    /tomas/{id}        toma completa
  POST   /tomas             guarda una toma (cuerpo: toma en JSON, puede venir comprimido con gzip)
  DELETE /tomas/{id}        borra una toma
  POST   /unir              { "ids": [...] } → une las tomas y guarda el resultado (espera a que termine)
  POST   /unir/iniciar      { "ids": [...] } → empieza la unión en segundo plano → { "trabajo": id }
  GET    /unir/{trabajo}    avance: { estado, avance (0–100), etapa, segundos, resultado? }

Sesiones con varias cámaras (el servidor arma un solo modelo con las vistas de todos los teléfonos):
  POST   /sesiones                          { nombre } → crea una sesión: { codigo }
  GET    /sesiones/{codigo}                 estado: nombre, cámaras, vistas, versión del modelo
  POST   /sesiones/{codigo}/camaras         { nombre } → { camara }
  POST   /sesiones/{codigo}/vistas          una vista: pose, imagen, máscara de la hoja → { vista }
  POST   /sesiones/{codigo}/vistas/{vista}/profundidad   su mapa de profundidad (relieve)
  GET    /sesiones/{codigo}/modelo?desde=N  modelo actual (forma + relieve + color) si cambió desde la versión N

Iniciar:  uvicorn server:app --host 0.0.0.0 --port $PORT
Variables de entorno: PORT (puerto), DATA_DIR (carpeta donde se guardan las tomas; por defecto ./data).
Ojo: en Render gratis el disco se borra cuando el servicio se reinicia. Para conservar las tomas, montar un
disco persistente y apuntar DATA_DIR a él. El teléfono igual guarda su propia copia y la vuelve a subir.
"""
import asyncio
import gzip
import threading
import uuid
import json
import os
import re
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse, Response
from starlette.concurrency import run_in_threadpool

from merge import merge_tomas
import sesion as S
from toma import summary, validate

DATA = Path(os.environ.get("DATA_DIR", Path.cwd() / "data"))
DATA.mkdir(parents=True, exist_ok=True)
MAX_BODY = 120 * 1024 * 1024

app = FastAPI(title="camara4d-backend")
app.add_middleware(GZipMiddleware, minimum_size=2048)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST", "HEAD", "DELETE", "OPTIONS"],
                   allow_headers=["Content-Type", "Content-Encoding"], max_age=86400)

# Índice en memoria: id → resumen (se arma al iniciar leyendo la carpeta)
index = {}
for f in DATA.glob("*.json.gz"):
    try:
        T = json.loads(gzip.decompress(f.read_bytes()))
        index[T["id"]] = summary(T)
    except Exception:
        pass   # archivo dañado

# Uniones de una en una (el cálculo usa mucha CPU y memoria)
merge_lock = asyncio.Lock()
work_lock = threading.Lock()

# Trabajos de unión en segundo plano: id → estado (se olvidan 1 hora después de terminar)
jobs = {}


def path_of(tid):
    return DATA / (re.sub(r"[^\w.-]", "_", tid) + ".json.gz")


def save(T):
    tmp = path_of(T["id"]).with_suffix(".tmp")
    tmp.write_bytes(gzip.compress(json.dumps(T, separators=(",", ":")).encode("utf-8"), compresslevel=5))
    tmp.replace(path_of(T["id"]))
    index[T["id"]] = summary(T)


def load(tid):
    return json.loads(gzip.decompress(path_of(tid).read_bytes()))


async def read_json(req: Request):
    body = await req.body()
    if len(body) > MAX_BODY:
        raise HTTPException(413, "toma demasiado grande")
    try:
        if req.headers.get("content-encoding") == "gzip":
            body = gzip.decompress(body)
        return json.loads(body)
    except Exception:
        raise HTTPException(400, "JSON inválido")


@app.exception_handler(HTTPException)
async def http_error(_, e: HTTPException):
    return JSONResponse({"error": e.detail}, status_code=e.status_code)


@app.exception_handler(ValueError)
async def value_error(_, e: ValueError):
    return JSONResponse({"error": str(e)}, status_code=400)


@app.get("/")
@app.get("/health")
def health():
    return {"ok": True, "servicio": "camara4d-backend", "tomas": len(index)}


@app.get("/tomas")
def lista():
    return sorted(index.values(), key=lambda s: s.get("fecha") or "", reverse=True)


@app.post("/tomas")
async def subir(req: Request):
    T = await read_json(req)
    validate(T)
    await run_in_threadpool(save, T)
    return {"ok": True, "id": T["id"]}


@app.head("/tomas/{tid}")
def existe(tid: str):
    return Response(status_code=200 if tid in index else 404)


@app.get("/tomas/{tid}")
async def obtener(tid: str):
    if tid not in index:
        raise HTTPException(404, "toma no encontrada")
    return await run_in_threadpool(load, tid)


@app.delete("/tomas/{tid}")
def borrar(tid: str):
    if tid not in index:
        raise HTTPException(404, "toma no encontrada")
    path_of(tid).unlink(missing_ok=True)
    index.pop(tid, None)
    return {"ok": True}


@app.post("/unir")
async def unir(req: Request):
    ids = _ids_validos(await read_json(req))
    async with merge_lock:
        await run_in_threadpool(work_lock.acquire)
        try:
            t0 = time.time()
            tomas = [await run_in_threadpool(load, i) for i in ids]
            r = await run_in_threadpool(merge_tomas, tomas)
            await run_in_threadpool(save, r["toma"])
            ms = round(1000 * (time.time() - t0))
        finally:
            work_lock.release()
    print("unir", ",".join(ids), "→", r["toma"]["id"], ms, "ms", flush=True)
    return {**r, "ms": ms}


# ---------- Sesiones con varias cámaras ----------
def _sesion(codigo):
    try:
        return S.buscar(codigo)
    except KeyError as e:
        raise HTTPException(404, str(e))


@app.post("/sesiones")
async def sesion_crear(req: Request):
    body = await read_json(req)
    s = S.crear(str(body.get("nombre", ""))[:60] if isinstance(body, dict) else "")
    return s.resumen()


@app.get("/sesiones/{codigo}")
def sesion_estado(codigo: str):
    return _sesion(codigo).resumen()


@app.post("/sesiones/{codigo}/camaras")
async def sesion_camara(codigo: str, req: Request):
    s = _sesion(codigo)
    body = await read_json(req)
    return {"camara": s.unir_camara(str(body.get("nombre", ""))[:40] if isinstance(body, dict) else ""), **s.resumen()}


@app.post("/sesiones/{codigo}/vistas")
async def sesion_vista(codigo: str, req: Request):
    s = _sesion(codigo)
    body = await read_json(req)
    vid = await run_in_threadpool(s.agregar_vista, body)
    return {"vista": vid, "version": s.version}


@app.post("/sesiones/{codigo}/vistas/{vid}/profundidad")
async def sesion_profundidad(codigo: str, vid: str, req: Request):
    s = _sesion(codigo)
    body = await read_json(req)
    await run_in_threadpool(s.agregar_profundidad, vid, body)
    return {"ok": True, "version": s.version}


@app.get("/sesiones/{codigo}/modelo")
async def sesion_modelo(codigo: str, desde: int = -1):
    s = _sesion(codigo)
    if desde >= 0 and s.version == desde:
        return {"version": s.version, "sin_cambios": True}
    return await run_in_threadpool(s.modelo)


def _ids_validos(body):
    ids = body.get("ids") if isinstance(body, dict) else None
    if not isinstance(ids, list) or len(ids) < 2:
        raise HTTPException(400, "se necesitan al menos 2 ids")
    faltan = [i for i in ids if i not in index]
    if faltan:
        raise HTTPException(404, "tomas no encontradas: " + ", ".join(map(str, faltan)))
    return ids


def _trabajo(jid, ids):
    J = jobs[jid]

    def avance(frac, etapa=None):
        J["avance"] = max(J["avance"], round(100 * min(1.0, max(0.0, frac)), 1))
        if etapa:
            J["etapa"] = etapa

    try:
        with work_lock:
            J["estado"] = "trabajando"
            J["etapa"] = "leyendo tomas"
            tomas = [load(i) for i in ids]
            r = merge_tomas(tomas, avance)
            save(r["toma"])
        J["resultado"] = {"id": r["toma"]["id"], "informe": r["informe"], "detalle": r["detalle"]}
        J["avance"], J["etapa"], J["estado"] = 100, "listo", "listo"
        print("unir", ",".join(ids), "→", r["toma"]["id"], round(time.time() - J["t0"], 1), "s", flush=True)
    except Exception as e:
        J["estado"], J["error"] = "error", str(e)
    J["fin"] = time.time()


@app.post("/unir/iniciar")
async def unir_iniciar(req: Request):
    ids = _ids_validos(await read_json(req))
    ahora = time.time()
    for k in [k for k, J in jobs.items() if J.get("fin") and ahora - J["fin"] > 3600]:
        jobs.pop(k, None)
    jid = uuid.uuid4().hex[:12]
    en_cola = sum(1 for J in jobs.values() if J["estado"] in ("en cola", "trabajando"))
    jobs[jid] = {"estado": "en cola", "avance": 0, "etapa": "esperando turno" if en_cola else "empezando", "t0": ahora, "ids": ids}
    threading.Thread(target=_trabajo, args=(jid, ids), daemon=True).start()
    return {"trabajo": jid}


@app.get("/unir/{jid}")
def unir_estado(jid: str):
    J = jobs.get(jid)
    if not J:
        raise HTTPException(404, "trabajo no encontrado (el servidor pudo reiniciarse)")
    out = {k: J[k] for k in ("estado", "avance", "etapa") if k in J}
    out["segundos"] = round(time.time() - J["t0"], 1)
    if "resultado" in J:
        out["resultado"] = J["resultado"]
    if "error" in J:
        out["error"] = J["error"]
    return out


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
  
