"""Extraer audio (MP3/WAV), transcribir, reconocer hablantes, detectar marcas de tiempo
por frase de inicio/fin y aislar una sola voz para entrenar modelos de voz.
Uso: python app.py  -> abre http://localhost:8765
Requiere ffmpeg en el PATH. Usa openai-whisper (o faster-whisper) + numpy + scipy.
"""
import io
import os
import re
import sys
import json
import uuid
import time
import wave
import shutil
import zipfile
import difflib
import tempfile
import threading
import subprocess
import unicodedata
import webbrowser
import urllib.parse
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler

import numpy as np
from scipy.fft import dct
from scipy.spatial.distance import cdist
from scipy.cluster.hierarchy import linkage, fcluster

PORT = 8765
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SALIDA = os.path.join(BASE_DIR, "audios")
TEMP_DIR = os.path.join(tempfile.gettempdir(), "extraer_audio_sesiones")
VOCES_REGISTRADAS_FILE = os.path.join(BASE_DIR, "voces_registradas.json")
os.makedirs(SALIDA, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

# Estado en memoria de trabajos y sesiones analizadas
JOBS_LOCK = threading.Lock()
JOBS = {}       # job_id -> dict con estado, progreso, mensaje, resultado, error
SESIONES = {}   # sesion_id -> dict con ruta_mp3, ruta_wav16k, nombre_base, duracion, segmentos, feats, etc.
WHISPER_CACHE = {}  # (motor, modelo) -> instancia cargada
WHISPER_LOCK = threading.Lock()
VOCES_LOCK = threading.Lock()
GIT_LOCK = threading.Lock()
GIT_ESTADO = {
    "disponible": False,
    "tiene_remote": False,
    "remote_url": "",
    "rama": "",
    "ahead": 0,
    "behind": 0,
    "estado": "verificando",
    "mensaje": "Verificando sincronización con GitHub…",
    "ultima_sinc": ""
}


# ==============================================================================
# SINCRONIZADOR AUTOMÁTICO DE GITHUB (PULL / PUSH AL INICIAR Y EN VIVO)
# ==============================================================================

def ejecutar_git(*args, timeout=25) -> tuple[int, str, str]:
    try:
        r = subprocess.run(
            ["git", "-C", BASE_DIR, *args],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout
        )
        return r.returncode, r.stdout.strip(), r.stderr.strip()
    except Exception as e:
        return -1, "", str(e)


def verificar_y_sincronizar_git(accion_automatica: bool = True) -> dict:
    """Verifica si el repositorio local está conectado a GitHub y realiza pull o push si corresponde."""
    with GIT_LOCK:
        code, out, _ = ejecutar_git("rev-parse", "--is-inside-work-tree")
        if code != 0:
            GIT_ESTADO.update({
                "disponible": False,
                "tiene_remote": False,
                "estado": "sin_git",
                "mensaje": "La carpeta no está inicializada con Git."
            })
            return dict(GIT_ESTADO)

        GIT_ESTADO["disponible"] = True

        code_b, rama, _ = ejecutar_git("branch", "--show-current")
        GIT_ESTADO["rama"] = rama or "main"

        code_rem, remote_url, _ = ejecutar_git("remote", "get-url", "origin")
        if code_rem != 0 or not remote_url:
            GIT_ESTADO.update({
                "tiene_remote": False,
                "remote_url": "",
                "estado": "sin_remote",
                "mensaje": "Repositorio local listo, pero aún no tiene un remoto 'origin' configurado en GitHub."
            })
            return dict(GIT_ESTADO)

        GIT_ESTADO["tiene_remote"] = True
        GIT_ESTADO["remote_url"] = remote_url

        # Hacer fetch para enterarnos de cambios en GitHub desde otra PC
        code_f, _, err_f = ejecutar_git("fetch", "origin", timeout=30)
        if code_f != 0:
            GIT_ESTADO.update({
                "estado": "error_fetch",
                "mensaje": f"No se pudo conectar a GitHub ({err_f[:120]}). Revisá conexión a internet o credenciales."
            })
            return dict(GIT_ESTADO)

        # Contar commits ahead (locales no subidos) y behind (en GitHub listos para pull)
        code_rev, counts, _ = ejecutar_git("rev-list", "--left-right", "--count", f"{GIT_ESTADO['rama']}...origin/{GIT_ESTADO['rama']}")
        ahead, behind = 0, 0
        if code_rev == 0 and counts:
            partes = counts.split()
            if len(partes) == 2:
                ahead, behind = int(partes[0]), int(partes[1])

        GIT_ESTADO["ahead"] = ahead
        GIT_ESTADO["behind"] = behind

        # Sincronización automática
        if accion_automatica:
            if behind > 0 and ahead == 0:
                # Caso ideal: hay cambios nuevos desde la otra PC -> hacer pull automático
                GIT_ESTADO["mensaje"] = f"Descargando {behind} cambios nuevos desde GitHub…"
                code_pull, out_p, err_p = ejecutar_git("pull", "--rebase", "origin", GIT_ESTADO["rama"])
                if code_pull == 0:
                    GIT_ESTADO["behind"] = 0
                    GIT_ESTADO["estado"] = "sincronizado"
                    GIT_ESTADO["mensaje"] = f"✔ Actualizado con éxito: se descargaron {behind} cambios desde la otra PC."
                    GIT_ESTADO["ultima_sinc"] = time.strftime("%H:%M:%S")
                else:
                    GIT_ESTADO["estado"] = "error_pull"
                    GIT_ESTADO["mensaje"] = f"Conflicto al hacer pull: {err_p[:120]}"
                return dict(GIT_ESTADO)

            elif ahead > 0 and behind == 0:
                # Hay cambios locales listos para subir a GitHub
                GIT_ESTADO["mensaje"] = f"Subiendo {ahead} cambios locales a GitHub…"
                code_push, _, err_push = ejecutar_git("push", "origin", GIT_ESTADO["rama"])
                if code_push == 0:
                    GIT_ESTADO["ahead"] = 0
                    GIT_ESTADO["estado"] = "sincronizado"
                    GIT_ESTADO["mensaje"] = f"✔ Sincronizado: se subieron {ahead} cambios a GitHub para tus otras PCs."
                    GIT_ESTADO["ultima_sinc"] = time.strftime("%H:%M:%S")
                else:
                    GIT_ESTADO["estado"] = "error_push"
                    GIT_ESTADO["mensaje"] = f"Error al hacer push: {err_push[:120]}"
                return dict(GIT_ESTADO)

            elif ahead > 0 and behind > 0:
                GIT_ESTADO["estado"] = "divergencia"
                GIT_ESTADO["mensaje"] = f"Hay {ahead} cambios locales y {behind} cambios en GitHub. Hacé clic en sincronizar."
                return dict(GIT_ESTADO)

        if ahead == 0 and behind == 0:
            GIT_ESTADO["estado"] = "sincronizado"
            GIT_ESTADO["mensaje"] = "✔ Todo al día. Sincronizado con GitHub."
            GIT_ESTADO["ultima_sinc"] = time.strftime("%H:%M:%S")

        return dict(GIT_ESTADO)


# ==============================================================================
# GESTIÓN DEL REGISTRO DE VOCES CONOCIDAS (HUELLAS VOCALES)
# ==============================================================================

def cargar_registro_voces() -> dict:
    with VOCES_LOCK:
        if not os.path.exists(VOCES_REGISTRADAS_FILE):
            return {"perfiles": {}}
        try:
            with open(VOCES_REGISTRADAS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                if not isinstance(data, dict):
                    data = {"perfiles": {}}
                data.setdefault("perfiles", {})
                return data
        except Exception:
            return {"perfiles": {}}


def guardar_registro_voces(data: dict) -> None:
    with VOCES_LOCK:
        try:
            with open(VOCES_REGISTRADAS_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Error guardando {VOCES_REGISTRADAS_FILE}: {e}")


def registrar_voz_perfil(nombre: str, vector_emb: np.ndarray, color: str = "") -> dict:
    """Guarda o actualiza la huella acústica de una persona en el archivo de registro."""
    nombre = nombre.strip()
    if not nombre:
        raise ValueError("El nombre no puede estar vacío")
    reg = cargar_registro_voces()
    perfiles = reg.setdefault("perfiles", {})

    emb_lista = [float(x) for x in vector_emb.flatten()]

    if nombre in perfiles and "embedding" in perfiles[nombre]:
        emb_ant = np.array(perfiles[nombre]["embedding"], dtype=np.float32)
        muestras_ant = perfiles[nombre].get("n_muestras", 1)
        # Promedio ponderado para refinar el perfil vocal a medida que se usa
        nuevo_emb = (emb_ant * muestras_ant + vector_emb) / float(muestras_ant + 1)
        perfiles[nombre] = {
            "nombre": nombre,
            "embedding": [float(x) for x in nuevo_emb.flatten()],
            "color": color or perfiles[nombre].get("color", "#2563eb"),
            "n_muestras": muestras_ant + 1,
            "actualizado_en": time.strftime("%Y-%m-%d %H:%M:%S")
        }
    else:
        perfiles[nombre] = {
            "nombre": nombre,
            "embedding": emb_lista,
            "color": color or "#2563eb",
            "n_muestras": 1,
            "creado_en": time.strftime("%Y-%m-%d %H:%M:%S")
        }

    guardar_registro_voces(reg)
    return perfiles[nombre]


def eliminar_voz_perfil(nombre: str) -> bool:
    reg = cargar_registro_voces()
    if nombre in reg.get("perfiles", {}):
        del reg["perfiles"][nombre]
        guardar_registro_voces(reg)
        return True
    return False


def emparejar_con_voces_registradas(centroides: np.ndarray, spk_ids: list[str], umbral_dist: float = 0.38) -> dict:
    """Compara cada centroide de hablante detectado contra los perfiles guardados.
    Retorna un diccionario {spk_id: {"nombre": ..., "color": ..., "distancia": ..., "coincidencia": True}}."""
    reg = cargar_registro_voces()
    perfiles = reg.get("perfiles", {})
    if not perfiles or len(centroides) == 0:
        return {}

    nombres_perfil = list(perfiles.keys())
    embs_guardados = np.array([perfiles[nom]["embedding"] for nom in nombres_perfil], dtype=np.float32)

    # Matriz de distancias coseno: filas = centroides detectados, columnas = perfiles guardados
    dist_mat = cdist(centroides, embs_guardados, metric="cosine")
    dist_mat = np.nan_to_num(dist_mat, nan=2.0)

    asignados = {}
    perfiles_usados = set()

    # Asignación voraz por menor distancia
    parejas = []
    for i, spk_id in enumerate(spk_ids):
        for j, nom in enumerate(nombres_perfil):
            parejas.append((dist_mat[i, j], i, j))
    parejas.sort(key=lambda x: x[0])

    for dist, i, j in parejas:
        spk_id = spk_ids[i]
        nom = nombres_perfil[j]
        if spk_id not in asignados and nom not in perfiles_usados:
            if dist <= umbral_dist:
                asignados[spk_id] = {
                    "nombre": nom,
                    "color": perfiles[nom].get("color", ""),
                    "distancia": round(float(dist), 3),
                    "auto_etiquetado": True
                }
                perfiles_usados.add(nom)

    return asignados


# ==============================================================================
# UTILIDADES DE TIEMPO, TEXTO Y ARCHIVOS
# ==============================================================================

def nombre_unico(carpeta: str, base: str, ext: str) -> str:
    """Devuelve un nombre de archivo que no pise uno existente en `carpeta`."""
    limpio = re.sub(r'[<>:"/\\|?*]', "_", base).strip() or "audio"
    nombre = f"{limpio}{ext}"
    i = 1
    while os.path.exists(os.path.join(carpeta, nombre)):
        nombre = f"{limpio} ({i}){ext}"
        i += 1
    return nombre


def parsear_tiempo(texto: str) -> float | None:
    """Convierte 'mm:ss', 'hh:mm:ss' o segundos a float."""
    texto = (texto or "").strip()
    if not texto:
        return None
    if re.fullmatch(r"\d+(\.\d+)?", texto):
        return float(texto)
    partes = texto.split(":")
    if len(partes) == 2 and all(p.isdigit() for p in partes):
        return int(partes[0]) * 60 + int(partes[1])
    if len(partes) == 3 and all(p.isdigit() for p in partes):
        return int(partes[0]) * 3600 + int(partes[1]) * 60 + int(partes[2])
    raise ValueError("Formato de tiempo inválido (usá mm:ss o hh:mm:ss)")


def normalizar_texto(t: str) -> str:
    """Quita tildes, signos de puntuación y pasa a minúsculas para buscar frases."""
    t = (t or "").lower()
    t = "".join(
        c for c in unicodedata.normalize("NFD", t)
        if unicodedata.category(c) != "Mn"
    )
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def buscar_frases_en_segmentos(segmentos: list[dict], frase: str, desde_seg: float = 0.0) -> list[dict]:
    """Busca `frase` en los segmentos transcritos (coincidencia exacta o aproximada)
    y calcula la marca de tiempo exacta donde empieza y termina la frase."""
    q = normalizar_texto(frase)
    if not q:
        return []
    palabras_q = q.split()
    n_q = len(palabras_q)
    coincidencias = []

    for idx, s in enumerate(segmentos):
        ini_s = float(s["inicio"])
        fin_s = float(s["fin"])
        if fin_s < desde_seg:
            continue
        dur_s = max(0.1, fin_s - ini_s)
        txt_norm = normalizar_texto(s.get("texto", ""))
        if not txt_norm:
            continue

        # 1. Búsqueda de subcadena directa en el segmento normalizado
        pos = txt_norm.find(q)
        if pos != -1:
            frac_ini = pos / max(1, len(txt_norm))
            frac_fin = (pos + len(q)) / max(1, len(txt_norm))
            t_frase_ini = round(ini_s + dur_s * frac_ini, 2)
            t_frase_fin = round(ini_s + dur_s * frac_fin, 2)
            coincidencias.append({
                "seg_id": idx,
                "seg_inicio": round(ini_s, 2),
                "seg_fin": round(fin_s, 2),
                "frase_inicio": t_frase_ini,
                "frase_fin": t_frase_fin,
                "texto": s.get("texto", ""),
                "hablante": s.get("hablante", "SPK_1"),
                "similitud": 1.0,
            })
            continue

        # 2. Búsqueda aproximada por ventana de palabras (por si Whisper cambió alguna letra/palabra)
        palabras_s = txt_norm.split()
        mejor_ratio = 0.0
        mejor_span = (0, len(palabras_s))
        if len(palabras_s) <= n_q:
            # También probar uniendo con el segmento siguiente por si la frase quedó partida entre dos segmentos
            txt_combo = txt_norm
            fin_combo = fin_s
            if idx + 1 < len(segmentos):
                txt_combo = (txt_norm + " " + normalizar_texto(segmentos[idx + 1].get("texto", ""))).strip()
                fin_combo = float(segmentos[idx + 1]["fin"])
            palabras_c = txt_combo.split()
            for i in range(max(1, len(palabras_c) - n_q + 1)):
                ventana = " ".join(palabras_c[i:i + n_q])
                r = difflib.SequenceMatcher(None, q, ventana).ratio()
                if r > mejor_ratio:
                    mejor_ratio = r
                    mejor_span = (i, min(len(palabras_c), i + n_q))
            if mejor_ratio >= 0.74:
                dur_c = max(0.1, fin_combo - ini_s)
                f_ini = mejor_span[0] / max(1, len(palabras_c))
                f_fin = mejor_span[1] / max(1, len(palabras_c))
                coincidencias.append({
                    "seg_id": idx,
                    "seg_inicio": round(ini_s, 2),
                    "seg_fin": round(fin_combo, 2),
                    "frase_inicio": round(ini_s + dur_c * f_ini, 2),
                    "frase_fin": round(ini_s + dur_c * f_fin, 2),
                    "texto": s.get("texto", ""),
                    "hablante": s.get("hablante", "SPK_1"),
                    "similitud": round(mejor_ratio, 2),
                })
        else:
            for i in range(len(palabras_s) - n_q + 1):
                ventana = " ".join(palabras_s[i:i + n_q])
                r = difflib.SequenceMatcher(None, q, ventana).ratio()
                if r > mejor_ratio:
                    mejor_ratio = r
                    mejor_span = (i, i + n_q)
            if mejor_ratio >= 0.74:
                f_ini = mejor_span[0] / max(1, len(palabras_s))
                f_fin = mejor_span[1] / max(1, len(palabras_s))
                coincidencias.append({
                    "seg_id": idx,
                    "seg_inicio": round(ini_s, 2),
                    "seg_fin": round(fin_s, 2),
                    "frase_inicio": round(ini_s + dur_s * f_ini, 2),
                    "frase_fin": round(ini_s + dur_s * f_fin, 2),
                    "texto": s.get("texto", ""),
                    "hablante": s.get("hablante", "SPK_1"),
                    "similitud": round(mejor_ratio, 2),
                })

    coincidencias.sort(key=lambda x: (-x["similitud"], x["frase_inicio"]))
    return coincidencias


def duracion_wav(ruta_wav: str) -> float:
    with wave.open(ruta_wav, "rb") as wf:
        return wf.getnframes() / float(wf.getframerate())


def extraer_mp3_ffmpeg(entrada: str, salida_mp3: str, cal: str = "192k",
                       desde: str = "", hasta: str = "", mb: float = 0.0,
                       mono: bool = False) -> None:
    """Extrae el audio en formato MP3 aplicando los recortes solicitados."""
    extra_in = []
    extra_out = []
    t_desde = 0.0
    if desde:
        t_desde = parsear_tiempo(desde) or 0.0
        extra_in += ["-ss", f"{t_desde:.2f}"]
    if hasta:
        t_hasta = parsear_tiempo(hasta)
        if t_hasta is not None and t_hasta > t_desde:
            extra_out += ["-t", f"{t_hasta - t_desde:.2f}"]
    if mb > 0:
        segundos = mb * 1024 * 1024 * 8 / (int(cal[:-1]) * 1000) * 0.98
        extra_out += ["-t", f"{segundos:.2f}"]
    if mono:
        extra_out += ["-ac", "1"]

    cmd = [
        "ffmpeg", "-y", *extra_in, "-i", entrada,
        "-vn", "-c:a", "libmp3lame", "-b:a", cal, *extra_out, salida_mp3
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0:
        raise RuntimeError("ffmpeg falló: " + r.stderr[-400:])


def recortar_video_ffmpeg(entrada: str, salida_mp4: str,
                          desde: str = "", hasta: str = "",
                          recodificar: bool = False) -> None:
    """Recorta un archivo de video entre `desde` (mm:ss / hh:mm:ss) y `hasta` (mm:ss / hh:mm:ss)."""
    extra_in = []
    extra_out = []
    t_desde = 0.0
    if desde:
        t_desde = parsear_tiempo(desde) or 0.0
        extra_in += ["-ss", f"{t_desde:.2f}"]
    if hasta:
        t_hasta = parsear_tiempo(hasta)
        if t_hasta is not None and t_hasta > t_desde:
            extra_out += ["-t", f"{t_hasta - t_desde:.2f}"]

    if not recodificar:
        cmd_copy = [
            "ffmpeg", "-y", *extra_in, "-i", entrada,
            *extra_out, "-c", "copy", "-movflags", "+faststart", salida_mp4
        ]
        r = subprocess.run(cmd_copy, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if r.returncode == 0 and os.path.exists(salida_mp4) and os.path.getsize(salida_mp4) > 1024:
            return

    cmd_enc = [
        "ffmpeg", "-y", *extra_in, "-i", entrada,
        *extra_out, "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
        "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", salida_mp4
    ]
    r2 = subprocess.run(cmd_enc, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r2.returncode != 0:
        raise RuntimeError("ffmpeg (recortar video) falló: " + r2.stderr[-400:])


def duracion_media(ruta: str) -> float:
    """Duración en segundos de un audio o video (ffprobe, o ffmpeg -i como respaldo)."""
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", ruta],
            capture_output=True, text=True, encoding="utf-8", errors="replace"
        )
        if r.returncode == 0 and r.stdout.strip():
            return float(r.stdout.strip().splitlines()[0])
    except (OSError, ValueError):
        pass
    r = subprocess.run(["ffmpeg", "-i", ruta], capture_output=True, text=True, encoding="utf-8", errors="replace")
    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", r.stderr or "")
    if not m:
        raise RuntimeError("No se pudo leer la duración del archivo.")
    return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))


def _fmt_tiempo_archivo(seg: float) -> str:
    seg = int(round(seg))
    h, m, s = seg // 3600, (seg % 3600) // 60, seg % 60
    return f"{h:02d}-{m:02d}-{s:02d}" if h else f"{m:02d}-{s:02d}"


def calcular_partes_split(duracion: float, modo: str, puntos_txt: str = "", dur_txt: str = "",
                          desde: str = "", hasta: str = "") -> list[tuple[float, float]]:
    """Devuelve la lista de partes (inicio, fin) en segundos.
    modo 'puntos': cortes en los minuto:segundo indicados (separados por coma, espacio o renglón).
    modo 'duracion': partes de duración fija (la última puede ser más corta).
    Si se indican desde/hasta, se divide solo ese tramo."""
    ini = parsear_tiempo(desde) or 0.0
    fin = parsear_tiempo(hasta) if hasta else None
    if fin is None or fin > duracion:
        fin = duracion
    if fin - ini <= 0.5:
        raise ValueError("El tramo a dividir es demasiado corto.")

    if modo == "puntos":
        crudos = [t for t in re.split(r"[\s,;]+", puntos_txt or "") if t]
        if not crudos:
            raise ValueError("Escribí al menos un punto de corte (ej. 10:00, 25:30).")
        puntos = sorted({parsear_tiempo(t) for t in crudos})
        puntos = [t for t in puntos if ini + 0.5 < t < fin - 0.5]
        if not puntos:
            raise ValueError("Ningún punto de corte cae dentro del archivo / tramo.")
        bordes = [ini] + puntos + [fin]
    elif modo == "duracion":
        paso = parsear_tiempo(dur_txt)
        if not paso or paso < 1:
            raise ValueError("Escribí la duración de cada parte (ej. 10:00 o 05:30).")
        if (fin - ini) / paso > 500:
            raise ValueError("Saldrían más de 500 partes: elegí una duración mayor.")
        bordes = [ini]
        while bordes[-1] + paso < fin - 0.5:
            bordes.append(bordes[-1] + paso)
        bordes.append(fin)
    else:
        raise ValueError("Modo de división inválido.")
    return [(bordes[i], bordes[i + 1]) for i in range(len(bordes) - 1)]


def _worker_dividir(job_id: str, ruta_entrada: str, base: str, params: dict) -> None:
    """Divide un video y/o audio en varias partes (por puntos de corte o duración fija)."""
    cal = params.get("cal", "192k")
    if cal not in ("64k", "128k", "192k", "320k"):
        cal = "192k"
    mono = bool(params.get("mono", False))
    modo_p1 = params.get("modo_p1", "mp3")
    exacto = bool(params.get("split_exacto", False))
    _actualizar_job(job_id, progreso=5, mensaje="Leyendo la duración del archivo…")
    duracion = duracion_media(ruta_entrada)
    partes = calcular_partes_split(
        duracion, params.get("split_modo", ""),
        puntos_txt=params.get("split_puntos", ""), dur_txt=params.get("split_duracion", ""),
        desde=params.get("desde", ""), hasta=params.get("hasta", "")
    )
    hacer_video = modo_p1 in ("video_mp4", "ambos")
    hacer_mp3 = modo_p1 in ("mp3", "ambos")
    total = len(partes)
    ancho = max(2, len(str(total)))
    resultado = []
    for i, (a, b) in enumerate(partes, start=1):
        _actualizar_job(job_id, progreso=5 + int(90 * (i - 1) / total),
                        mensaje=f"Parte {i} de {total} ({_fmt_tiempo_archivo(a).replace('-', ':')} → {_fmt_tiempo_archivo(b).replace('-', ':')})…")
        etiqueta = f"{base}_parte_{i:0{ancho}d}_{_fmt_tiempo_archivo(a)}_a_{_fmt_tiempo_archivo(b)}"
        item = {"n": i, "inicio": a, "fin": b, "nombre_video": "", "nombre_mp3": ""}
        if hacer_video:
            nv = nombre_unico(SALIDA, etiqueta, ".mp4")
            recortar_video_ffmpeg(ruta_entrada, os.path.join(SALIDA, nv),
                                  desde=f"{a:.2f}", hasta=f"{b:.2f}", recodificar=exacto)
            item["nombre_video"] = nv
        if hacer_mp3:
            na = nombre_unico(SALIDA, etiqueta, ".mp3")
            extraer_mp3_ffmpeg(ruta_entrada, os.path.join(SALIDA, na), cal=cal,
                               desde=f"{a:.2f}", hasta=f"{b:.2f}", mono=mono)
            item["nombre_mp3"] = na
        resultado.append(item)
    _actualizar_job(job_id, estado="completado", progreso=100,
                    mensaje=f"Listo: {total} partes.",
                    resultado={"modo": "partes", "partes": resultado, "duracion": duracion})


# ==============================================================================
# VIDEO PARA PUBLICAR: SILENCIAR VOCES ELEGIDAS + SUBTÍTULOS DE LO QUE DIJERON
# ==============================================================================
def _tiene_pista_audio(ruta: str) -> bool:
    try:
        r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries",
                            "stream=index", "-of", "csv=p=0", ruta],
                           capture_output=True, text=True, encoding="utf-8", errors="replace")
        if r.returncode == 0:
            return bool(r.stdout.strip())
    except OSError:
        pass
    r = subprocess.run(["ffmpeg", "-i", ruta], capture_output=True, text=True, encoding="utf-8", errors="replace")
    return "Audio:" in (r.stderr or "")


def _srt_desde_bloques(bloques: list[dict]) -> str:
    def ts(seg):
        ms = int(round(seg * 1000))
        return f"{ms//3600000:02d}:{(ms%3600000)//60000:02d}:{(ms%60000)//1000:02d},{ms%1000:03d}"
    return "\n".join(f"{i}\n{ts(b['ini'])} --> {ts(b['fin'])}\n{b['texto_srt']}\n"
                     for i, b in enumerate(bloques, start=1))


def _worker_publicar(job_id: str, params: dict) -> None:
    """Genera un MP4 donde las voces elegidas quedan en silencio y lo que dijeron aparece
    como pista de subtítulos (además de un .SRT aparte). Opcionalmente nivela el audio."""
    ruta_script = ""
    ruta_srt_tmp = ""
    try:
        sid = params["sesion_id"]
        with JOBS_LOCK:
            sesion = SESIONES.get(sid)
        if not sesion:
            raise ValueError("No se encontró el análisis de voces. Volvé a analizar el video.")
        nombre_video = os.path.basename(params["nombre_video"])
        ruta_video = os.path.join(SALIDA, nombre_video)
        if not os.path.isfile(ruta_video):
            raise ValueError(f"No se encontró el video en audios/: {nombre_video}")

        silenciados = set(params.get("silenciados") or [])
        solo_sub = set(params.get("subtitular") or []) - silenciados
        offset = float(params.get("offset", 0) or 0)
        margen = max(0.0, min(1.0, float(params.get("margen", 0.2))))
        nivelar = bool(params.get("nivelar", True))
        igualar = bool(params.get("igualar", False))
        modo_nombre = params.get("modo_nombre", "anon")
        prefijo = (params.get("prefijo_anon") or "Alumno").strip() or "Alumno"
        nombres_edit = params.get("nombres") or {}
        tiene_audio = _tiene_pista_audio(ruta_video)
        if silenciados and not tiene_audio:
            raise ValueError("El video no tiene audio: no hay voces para silenciar. Usá «Solo subtítulo».")

        segmentos = sesion.get("segmentos", [])
        a_silenciar = [s for s in segmentos if s.get("hablante") in silenciados]
        a_subtitular = [s for s in segmentos if s.get("hablante") in silenciados or s.get("hablante") in solo_sub]
        if not a_subtitular and not ((igualar or nivelar) and tiene_audio):
            raise ValueError("Elegí al menos una voz para silenciar o subtitular, o activá igualar / nivelar el volumen.")

        # Nombres que se muestran en los subtítulos
        orden_spk = []
        for s in sorted(a_subtitular, key=lambda x: float(x.get("inicio", 0))):
            if s.get("hablante") not in orden_spk:
                orden_spk.append(s.get("hablante"))
        mapa_real = {h["id"]: (nombres_edit.get(h["id"]) or h.get("nombre") or h["id"]) for h in sesion.get("hablantes", [])}
        if modo_nombre == "real":
            mapa_nom = {spk: mapa_real.get(spk, spk) for spk in orden_spk}
        elif modo_nombre == "ninguno":
            mapa_nom = {spk: "" for spk in orden_spk}
        else:
            mapa_nom = {spk: f"{prefijo} {i}" for i, spk in enumerate(orden_spk, start=1)}

        _actualizar_job(job_id, progreso=4, mensaje="Armando los subtítulos de las voces silenciadas…")
        segs_sub = [{k: v for k, v in s.items() if k != "override_mute"} for s in a_subtitular]
        bloques = construir_subtitulos_profesionales(
            segs_sub, mapa_nom=mapa_nom, mapa_sil={}, offset_seg=offset, inc_nom=(modo_nombre != "ninguno"),
            inc_sil=True, agregar_bloque_00=False,
            max_chars_sub=int(params.get("max_chars", 84) or 84), min_dur_seg=1.2,
            max_dur_seg=float(params.get("max_dur", 6.0) or 6.0),
        )
        texto_srt = _srt_desde_bloques(bloques) if bloques else ""
        base = os.path.splitext(nombre_video)[0] + "_publicar"
        nombre_srt = ""
        if bloques:
            nombre_srt = nombre_unico(SALIDA, base, ".srt")
            with open(os.path.join(SALIDA, nombre_srt), "w", encoding="utf-8") as f:
                f.write(texto_srt)

        # Filtro de audio: silencio en los tramos + nivelación opcional
        intervalos = _unir_intervalos(
            [(max(0.0, float(s["inicio"]) + offset - margen), float(s["fin"]) + offset + margen)
             for s in a_silenciar if float(s["fin"]) + offset > 0],
            margen_union=0.3) if a_silenciar else []
        filtros = []
        if intervalos:
            expr = "+".join(f"between(t,{a:.2f},{b:.2f})" for a, b in intervalos)
            filtros.append(f"volume=enable='{expr}':volume=0")
        if igualar and tiene_audio:
            # Normalizador dinámico: sube las voces bajas y baja las altas (sin amplificar el silencio de más)
            filtros.append("dynaudnorm=f=300:g=15:p=0.9:m=8:s=12")
        if nivelar and tiene_audio:
            filtros += ["loudnorm=I=-16:TP=-1.5:LRA=11", "aresample=48000"]
        filtro = ",".join(filtros)

        nombre_out = nombre_unico(SALIDA, base, ".mp4")
        ruta_out = os.path.join(SALIDA, nombre_out)
        ruta_srt_tmp = os.path.join(TEMP_DIR, f"pub_{uuid.uuid4().hex[:8]}.srt")
        with open(ruta_srt_tmp, "w", encoding="utf-8") as f:
            f.write(texto_srt)
        ruta_script = os.path.join(TEMP_DIR, f"pub_{uuid.uuid4().hex[:8]}_filtro.txt")
        with open(ruta_script, "w", encoding="utf-8") as f:
            f.write(filtro)

        try:
            duracion = duracion_media(ruta_video)
        except Exception:
            duracion = float(sesion.get("duracion") or 0) or 1.0

        def correr(opciones_filtro: list[str], codec_video: list[str]) -> tuple[int, str]:
            mapa_audio = ["-map", "0:a:0", "-c:a", "aac", "-b:a", "192k"] if tiene_audio else []
            subs_in = ["-i", ruta_srt_tmp] if bloques else []
            subs_map = ["-map", "1:0"] if bloques else []
            subs_opts = (["-c:s", "mov_text", "-metadata:s:s:0", "language=spa",
                          "-metadata:s:s:0", "title=Subtitulos", "-disposition:s:0", "default"] if bloques else [])
            cmd = ["ffmpeg", "-y", "-i", ruta_video, *subs_in,
                   "-map", "0:v:0", *mapa_audio[:2], *subs_map,
                   *opciones_filtro, *codec_video, *mapa_audio[2:], *subs_opts,
                   "-movflags", "+faststart", "-progress", "pipe:1", "-nostats", ruta_out]
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                    text=True, encoding="utf-8", errors="replace")
            err_buf = []
            hilo_err = threading.Thread(target=lambda: err_buf.append(proc.stderr.read()), daemon=True)
            hilo_err.start()
            for linea in proc.stdout:
                if linea.startswith("out_time_us=") or linea.startswith("out_time_ms="):
                    try:
                        seg = int(linea.split("=", 1)[1]) / 1_000_000
                        _actualizar_job(job_id, progreso=min(97, 8 + int(89 * seg / max(1.0, duracion))),
                                        mensaje=f"Generando el video… {fmt_hms_simple(seg)} de {fmt_hms_simple(duracion)}")
                    except ValueError:
                        pass
            proc.wait()
            hilo_err.join(timeout=5)
            return proc.returncode, "".join(err_buf)

        variantes_filtro = []
        if not filtro:
            variantes_filtro.append([])
        elif len(filtro) < 12000:
            variantes_filtro.append(["-af", filtro])
        if filtro:
            variantes_filtro += [["-filter_script:a", ruta_script], ["-/af", ruta_script]]

        _actualizar_job(job_id, progreso=8, mensaje="Silenciando voces y uniendo subtítulos…" if intervalos else "Uniendo subtítulos…")
        ultimo_err = ""
        hecho = False
        for codec_video in (["-c:v", "copy"], ["-c:v", "libx264", "-preset", "veryfast", "-crf", "23"]):
            for vf in variantes_filtro:
                rc, err = correr(vf, codec_video)
                if rc == 0 and os.path.exists(ruta_out) and os.path.getsize(ruta_out) > 1024:
                    hecho = True
                    break
                ultimo_err = err
                # Si el error no es por la opción de filtro, no tiene sentido probar otra variante de filtro
                if "Unrecognized option" not in err and "Option not found" not in err:
                    break
            if hecho:
                break
        if not hecho:
            raise RuntimeError("ffmpeg falló: " + (ultimo_err or "")[-500:])

        dur_sil = sum(b - a for a, b in intervalos)
        _actualizar_job(job_id, estado="completado", progreso=100, mensaje="Video listo.", resultado={
            "nombre_video": nombre_out,
            "nombre_srt": nombre_srt,
            "tramos_silenciados": len(intervalos),
            "segundos_silenciados": round(dur_sil, 1),
            "subtitulos": len(bloques),
            "etiquetas": mapa_nom,
            "nivelado": nivelar and tiene_audio,
            "igualado": igualar and tiene_audio,
            "voces_silenciadas": len(silenciados),
            "voces_solo_sub": len(solo_sub),
        })
    except Exception as e:
        _actualizar_job(job_id, estado="error", mensaje=str(e), error=str(e))
    finally:
        for rp in (ruta_script, ruta_srt_tmp):
            if rp and os.path.exists(rp):
                try:
                    os.remove(rp)
                except OSError:
                    pass


def fmt_hms_simple(seg: float) -> str:
    seg = int(max(0, seg))
    h, m, s = seg // 3600, (seg % 3600) // 60, seg % 60
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def _dividir_texto_en_frases_sub(texto: str, max_chars: int = 84) -> list[str]:
    """Divide un texto largo en fragmentos aptos para subtítulos (<= max_chars),
    priorizando cortes en signos de puntuación (. ? ! ; : ,) y nunca cortando palabras."""
    texto = re.sub(r"https?://\S{28,}", lambda m: m.group(0)[:24] + "…", (texto or ""))
    texto = re.sub(r"\s+", " ", texto.strip())
    if not texto:
        return []
    if len(texto) <= max_chars:
        return [texto]

    partes_punt = re.split(r"(?<=[.?!;:])\s+", texto)
    fragmentos = []
    actual = ""

    def _empujar_palabras(bloque_txt: str):
        nonlocal actual
        for pal in bloque_txt.split(" "):
            if not pal:
                continue
            if len(pal) > max_chars:
                pal = pal[:max_chars - 1] + "…"
            if not actual:
                actual = pal
            elif len(actual) + 1 + len(pal) <= max_chars:
                actual += " " + pal
            else:
                fragmentos.append(actual)
                actual = pal

    for p in partes_punt:
        p = p.strip()
        if not p:
            continue
        if not actual and len(p) <= max_chars:
            actual = p
        elif actual and len(actual) + 1 + len(p) <= max_chars:
            actual += " " + p
        else:
            if actual:
                fragmentos.append(actual)
                actual = ""
            if len(p) <= max_chars:
                actual = p
            else:
                for sc in re.split(r"(?<=,)\s+", p):
                    sc = sc.strip()
                    if not sc:
                        continue
                    if not actual and len(sc) <= max_chars:
                        actual = sc
                    elif actual and len(actual) + 1 + len(sc) <= max_chars:
                        actual += " " + sc
                    else:
                        if actual:
                            fragmentos.append(actual)
                            actual = ""
                        if len(sc) <= max_chars:
                            actual = sc
                        else:
                            _empujar_palabras(sc)

    if actual:
        fragmentos.append(actual)
    return fragmentos


def _balancear_dos_lineas_sub(texto: str, max_por_linea: int = 42) -> str:
    """Si el subtítulo supera max_por_linea caracteres, lo divide en 2 líneas equilibradas."""
    t = re.sub(r"\s+", " ", (texto or "").strip())
    if len(t) <= max_por_linea or " " not in t:
        return t
    mitad = len(t) // 2
    mejor_pos = -1
    mejor_dist = 10**9
    for m in re.finditer(r" ", t):
        pos = m.start()
        dist = abs(pos - mitad)
        if dist < mejor_dist:
            mejor_dist = dist
            mejor_pos = pos
    if mejor_pos <= 0:
        return t
    return t[:mejor_pos].strip() + "\n" + t[mejor_pos + 1:].strip()


def construir_subtitulos_profesionales(
    segmentos: list[dict],
    mapa_nom: dict,
    mapa_sil: dict,
    offset_seg: float = 0.0,
    inc_nom: bool = True,
    inc_sil: bool = False,
    agregar_bloque_00: bool = True,
    t_desde_v: float | None = None,
    t_hasta_v: float | None = None,
    max_chars_sub: int = 84,
    min_dur_seg: float = 1.2,
    max_dur_seg: float = 6.0,
) -> list[dict]:
    """Construye bloques de subtítulos profesionales garantizando:
    1. CERO superposiciones entre subtítulos (separación mínima de 40 ms).
    2. Largo máximo de caracteres por subtítulo (<= max_chars_sub, dividido en hasta 2 líneas balanceadas).
    3. Duración en pantalla legible y controlada (mínimo 1.2s y máximo max_dur_seg por cartel).
    """
    GAP = 0.04
    VEL_CHARS_SEG = 15.0
    max_chars_sub = max(40, min(140, int(max_chars_sub or 84)))
    max_linea = max(25, max_chars_sub // 2)
    min_dur_seg = max(1.0, min(3.5, float(min_dur_seg or 1.2)))
    max_dur_seg = max(min_dur_seg + 1.0, min(12.0, float(max_dur_seg or 6.0)))
    t0_shift = float(t_desde_v) if t_desde_v is not None and t_desde_v > 0 else 0.0

    crudos = []
    for s in segmentos:
        sil = s.get("override_mute") if s.get("override_mute") is not None else mapa_sil.get(s.get("hablante"), False)
        if sil and not inc_sil:
            continue
        ini_real = float(s.get("inicio", 0.0)) + offset_seg
        fin_real = float(s.get("fin", 0.0)) + offset_seg
        if fin_real <= 0.05:
            continue
        ini_abs = max(0.0, ini_real)
        fin_abs = max(ini_abs + 0.4, fin_real)
        if t_desde_v is not None and fin_abs <= t_desde_v:
            continue
        if t_hasta_v is not None and ini_abs >= t_hasta_v:
            continue
        ini = max(0.0, ini_abs - t0_shift)
        fin = max(ini + 0.4, (min(t_hasta_v, fin_abs) if t_hasta_v is not None else fin_abs) - t0_shift)
        nom_spk = mapa_nom.get(s.get("hablante"), s.get("hablante", "Hablante"))
        crudos.append({
            "ini": ini,
            "fin": fin,
            "ini_orig": float(s.get("inicio", 0.0)),
            "hablante": nom_spk,
            "texto": (s.get("texto") or "").strip(),
        })

    crudos.sort(key=lambda x: (x["ini"], x["fin"]))
    n_c = len(crudos)
    bloques_finales = []
    cursor_global = 0.0

    for idx_c, c in enumerate(crudos):
        pref = f"[{c['hablante']}] " if (inc_nom and c["hablante"]) else ""
        presupuesto_txt = max(20, max_chars_sub - len(pref))
        partes_txt = _dividir_texto_en_frases_sub(c["texto"], presupuesto_txt)
        if not partes_txt:
            continue

        m_partes = len(partes_txt)
        ini_seg = max(c["ini"], cursor_global)

        # Duración ideal de cada sub-bloque según velocidad de lectura (~15 chars/s), acotada a [min_dur_seg, max_dur_seg]
        durs_ideales = [
            max(min_dur_seg, min(max_dur_seg, len(pref + pt) / VEL_CHARS_SEG))
            for pt in partes_txt
        ]
        suma_ideal = sum(durs_ideales)

        # Si hay un siguiente segmento y su inicio permite acomodar los carteles sin bajar de min_dur_seg, ajustamos al hueco
        if idx_c + 1 < n_c:
            prox_ini = crudos[idx_c + 1]["ini"]
            hueco_neto = (prox_ini - ini_seg) - m_partes * GAP
        else:
            hueco_neto = max(suma_ideal, c["fin"] - ini_seg)

        dur_orig_neta = max(suma_ideal, min(c["fin"] - ini_seg, m_partes * max_dur_seg))
        if hueco_neto >= m_partes * min_dur_seg:
            dur_total_usar = min(dur_orig_neta, hueco_neto)
        else:
            # Si dos marcas en Tactiq estaban demasiado pegadas para la cantidad de texto,
            # garantizamos al menos min_dur_seg por cartel para que siempre se pueda leer en pantalla
            dur_total_usar = max(m_partes * min_dur_seg, min(suma_ideal, m_partes * max_dur_seg))

        cursor_sub = ini_seg
        for idx_p, pt in enumerate(partes_txt):
            proporcion = durs_ideales[idx_p] / max(0.1, suma_ideal)
            dur_p = max(min_dur_seg, min(max_dur_seg, dur_total_usar * proporcion))
            ini_b = round(cursor_sub, 3)
            fin_b = round(ini_b + dur_p, 3)
            txt_completo = pref + pt
            bloques_finales.append({
                "ini": ini_b,
                "fin": fin_b,
                "ini_orig": c["ini_orig"],
                "hablante": c["hablante"],
                "es_bloque_cero": False,
                "texto_plano": txt_completo,
                "texto_srt": _balancear_dos_lineas_sub(txt_completo, max_linea),
            })
            cursor_sub = round(fin_b + GAP, 3)

        cursor_global = cursor_sub

    if agregar_bloque_00 and bloques_finales and bloques_finales[0]["ini"] >= 0.5:
        s0 = int(round(bloques_finales[0]["ini"]))
        ts0 = f"{s0//60:02d}:{s0%60:02d}" if s0 < 3600 else f"{s0//3600:02d}:{(s0%3600)//60:02d}:{s0%60:02d}"
        t_primero = round(bloques_finales[0]["ini"] - GAP, 3)
        msg_00 = f"[Inicio del video 00:00 — La transcripción comienza en {ts0}]"
        bloques_finales.insert(0, {
            "ini": 0.0,
            "fin": max(0.4, t_primero),
            "ini_orig": None,
            "hablante": "",
            "es_bloque_cero": True,
            "texto_plano": msg_00,
            "texto_srt": _balancear_dos_lineas_sub(msg_00, max_linea),
        })

    return bloques_finales


def recortar_segmentos_por_tiempo(segmentos: list[dict], desde_seg: float | None,
                                  hasta_seg: float | None,
                                  reiniciar_reloj: bool = False) -> tuple[list[dict], list[int], float]:
    """Filtra y recorta los segmentos de transcripción entre [desde_seg, hasta_seg] (en segundos).
    Devuelve (nuevos_segmentos, indices_originales, nueva_duracion)."""
    t0 = float(desde_seg) if desde_seg is not None and desde_seg > 0 else 0.0
    t1 = float(hasta_seg) if hasta_seg is not None and hasta_seg > 0 else None

    nuevos = []
    indices = []
    for idx, s in enumerate(segmentos):
        ini = float(s.get("inicio", 0.0))
        fin = float(s.get("fin", ini))
        if fin <= t0:
            continue
        if t1 is not None and ini >= t1:
            continue
        n_ini = max(t0, ini)
        n_fin = min(t1, fin) if t1 is not None else fin
        if n_fin <= n_ini:
            continue
        if reiniciar_reloj and t0 > 0:
            n_ini = max(0.0, n_ini - t0)
            n_fin = max(n_ini + 0.1, n_fin - t0)
        copia = dict(s)
        copia["id"] = len(nuevos)
        copia["inicio"] = round(n_ini, 2)
        copia["fin"] = round(n_fin, 2)
        nuevos.append(copia)
        indices.append(idx)

    dur = round(max((float(x["fin"]) for x in nuevos), default=0.0), 2)
    return nuevos, indices, dur


def convertir_a_wav16k(entrada_audio: str, salida_wav: str) -> None:
    """Convierte cualquier audio/video a WAV mono 16kHz PCM 16-bit para Whisper y análisis de voz."""
    cmd = [
        "ffmpeg", "-y", "-i", entrada_audio,
        "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", salida_wav
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0:
        raise RuntimeError("ffmpeg (wav16k) falló: " + r.stderr[-400:])


# ==============================================================================
# EXTRACCIÓN DE HUELLA ACÚSTICA Y DIARIZACIÓN DE HABLANTES (100% LOCAL)
# ==============================================================================

def _banco_mel(n_mels: int, n_fft: int, sr: int = 16000, fmin: float = 80.0, fmax: float = 7600.0) -> np.ndarray:
    def hz_a_mel(hz):
        return 2595.0 * np.log10(1.0 + hz / 700.0)

    def mel_a_hz(mel):
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    mels = np.linspace(hz_a_mel(fmin), hz_a_mel(fmax), n_mels + 2)
    hz = mel_a_hz(mels)
    bins = np.floor((n_fft + 1) * hz / sr).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for m in range(1, n_mels + 1):
        f_izq, f_cen, f_der = bins[m - 1], bins[m], bins[m + 1]
        if f_cen == f_izq:
            f_cen += 1
        if f_der == f_cen:
            f_der += 1
        for k in range(f_izq, f_cen):
            if 0 <= k < fb.shape[1]:
                fb[m - 1, k] = (k - f_izq) / (f_cen - f_izq)
        for k in range(f_cen, f_der):
            if 0 <= k < fb.shape[1]:
                fb[m - 1, k] = (f_der - k) / (f_der - f_cen)
    return fb


MEL_FB_16K = _banco_mel(32, 512, 16000)


def cargar_audio_16k(ruta_wav: str) -> np.ndarray:
    with wave.open(ruta_wav, "rb") as wf:
        raw = wf.readframes(wf.getnframes())
    pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    return pcm


def subdividir_segmentos_largos(segmentos: list[dict], wav: np.ndarray, sr: int = 16000) -> list[dict]:
    resultado = []
    for seg in segmentos:
        ini, fin, txt = float(seg["inicio"]), float(seg["fin"]), seg["texto"].strip()
        dur = fin - ini
        if dur <= 6.5:
            resultado.append({"inicio": round(ini, 2), "fin": round(fin, 2), "texto": txt})
            continue
        partes_txt = [p.strip() for p in re.split(r'(?<=[.?!])\s+', txt) if p.strip()]
        if len(partes_txt) <= 1:
            resultado.append({"inicio": round(ini, 2), "fin": round(fin, 2), "texto": txt})
            continue
        total_chars = sum(len(p) for p in partes_txt)
        cursor_t = ini
        for idx, p_txt in enumerate(partes_txt):
            if idx == len(partes_txt) - 1:
                t_fin = fin
            else:
                prop = len(p_txt) / max(1, total_chars)
                t_est = cursor_t + dur * prop
                w_ini = max(int((t_est - 0.35) * sr), int((cursor_t + 0.4) * sr))
                w_fin = min(int((t_est + 0.35) * sr), int((fin - 0.4) * sr))
                if w_fin > w_ini + 800:
                    ventana = wav[w_ini:w_fin]
                    paso = 400
                    n_pasos = (len(ventana) - paso) // paso
                    if n_pasos > 0:
                        energias = [np.mean(ventana[i * paso:(i + 1) * paso] ** 2) for i in range(n_pasos)]
                        mejor_i = int(np.argmin(energias))
                        t_fin = (w_ini + mejor_i * paso) / float(sr)
                    else:
                        t_fin = t_est
                else:
                    t_fin = t_est
            resultado.append({"inicio": round(cursor_t, 2), "fin": round(t_fin, 2), "texto": p_txt})
            cursor_t = t_fin
    return resultado


def extraer_features_segmento(trozo: np.ndarray, sr: int = 16000) -> np.ndarray:
    n_win = 400
    hop = 160
    n_fft = 512
    if len(trozo) < n_win * 2:
        trozo = np.pad(trozo, (0, n_win * 2 - len(trozo)))

    trozo_pe = np.append(trozo[0], trozo[1:] - 0.97 * trozo[:-1])

    n_frames = 1 + (len(trozo_pe) - n_win) // hop
    strides = (trozo_pe.strides[0] * hop, trozo_pe.strides[0])
    frames = np.lib.stride_tricks.as_strided(trozo_pe, shape=(n_frames, n_win), strides=strides)
    frames_raw = np.lib.stride_tricks.as_strided(trozo, shape=(n_frames, n_win), strides=(trozo.strides[0] * hop, trozo.strides[0]))

    energia = np.mean(frames_raw ** 2, axis=1)
    umbral = max(1e-7, np.percentile(energia, 30))
    activos = energia >= umbral
    if np.sum(activos) < 3:
        activos = np.ones(n_frames, dtype=bool)

    ventana = np.hanning(n_win).astype(np.float32)
    espectro = np.abs(np.fft.rfft(frames[activos] * ventana, n=n_fft, axis=1)) ** 2

    mel_e = np.dot(espectro, MEL_FB_16K.T)
    log_mel = np.log(np.maximum(mel_e, 1e-10))
    mfccs = dct(log_mel, type=2, axis=1, norm="ortho")[:, 1:21]
    mfcc_mean = np.mean(mfccs, axis=0)
    mfcc_std = np.std(mfccs, axis=0)

    freqs = np.linspace(0, sr / 2.0, espectro.shape[1])
    suma_esp = np.maximum(np.sum(espectro, axis=1), 1e-9)
    centroide = np.mean(np.sum(espectro * freqs, axis=1) / suma_esp) / 4000.0
    acum = np.cumsum(espectro, axis=1)
    idx_roll = np.argmax(acum >= 0.85 * suma_esp[:, None], axis=1)
    rolloff = np.mean(freqs[idx_roll]) / 6000.0

    lag_min = int(sr / 400.0)
    lag_max = int(sr / 75.0)
    f0_vals = []
    f_sonoros = frames_raw[activos]
    paso_f = max(1, len(f_sonoros) // 60)
    for fr in f_sonoros[::paso_f]:
        fr0 = fr - np.mean(fr)
        e0 = np.sum(fr0 * fr0)
        if e0 < 1e-6:
            continue
        corr = np.correlate(fr0, fr0, mode="full")[len(fr0) - 1:]
        if len(corr) > lag_max:
            sub = corr[lag_min:lag_max] / e0
            pico = int(np.argmax(sub))
            if sub[pico] > 0.32:
                f0 = sr / float(lag_min + pico)
                f0_vals.append(np.log(f0))

    if len(f0_vals) >= 2:
        f0_med = float(np.median(f0_vals))
        f0_iqr = float(np.percentile(f0_vals, 75) - np.percentile(f0_vals, 25))
    elif len(f0_vals) == 1:
        f0_med = float(f0_vals[0])
        f0_iqr = 0.05
    else:
        f0_med = np.log(160.0)
        f0_iqr = 0.1

    return np.concatenate([
        mfcc_mean,
        mfcc_std * 0.35,
        [f0_med * 3.2, f0_iqr * 1.5, centroide * 1.2, rolloff * 1.2]
    ]).astype(np.float32)


def calcular_matriz_features(segmentos: list[dict], wav: np.ndarray, sr: int = 16000) -> np.ndarray:
    if not segmentos:
        return np.zeros((0, 44), dtype=np.float32)
    feats = []
    for s in segmentos:
        i0 = max(0, int(float(s["inicio"]) * sr))
        i1 = min(len(wav), int(float(s["fin"]) * sr))
        feats.append(extraer_features_segmento(wav[i0:i1], sr))
    X = np.vstack(feats)
    if len(X) > 1:
        med = np.median(X, axis=0)
        mad = np.median(np.abs(X - med), axis=0) * 1.4826
        mad = np.where(mad < 1e-4, 1.0, mad)
        X = (X - med) / mad
        pesos = np.ones(X.shape[1], dtype=np.float32)
        pesos[0:20] = 1.0
        pesos[20:40] = 0.30
        pesos[40] = 2.4
        pesos[41] = 0.8
        pesos[42:44] = 0.7
        X = X * pesos
    return X


def _silhouette_score_coseno(X: np.ndarray, labels: np.ndarray) -> float:
    unicos = np.unique(labels)
    if len(unicos) < 2 or len(X) <= len(unicos):
        return -1.0
    D = cdist(X, X, metric="cosine")
    D = np.nan_to_num(D, nan=1.0)
    scores = []
    for i in range(len(X)):
        c_propio = labels[i]
        mismo = (labels == c_propio)
        mismo[i] = False
        if np.sum(mismo) == 0:
            a_i = 0.0
        else:
            a_i = float(np.mean(D[i, mismo]))
        b_i = min(float(np.mean(D[i, labels == c])) for c in unicos if c != c_propio)
        denom = max(a_i, b_i, 1e-6)
        scores.append((b_i - a_i) / denom)
    return float(np.mean(scores))


def agrupar_hablantes(segmentos: list[dict], X: np.ndarray | None, num_hablantes: int = 0, **kwargs) -> tuple[list[dict], list[dict]]:
    n = len(segmentos)
    if n == 0:
        return [], []
    if n == 1 or num_hablantes == 1 or X is None or len(X) != n:
        for idx, s in enumerate(segmentos):
            s["id"] = idx
            s.setdefault("hablante", "SPK_1")
            s.setdefault("confianza", 1.0)
            s.setdefault("silenciado", False)
        return segmentos, _resumen_hablantes(segmentos)

    duraciones = np.array([max(0.1, float(s["fin"]) - float(s["inicio"])) for s in segmentos])
    anclas = duraciones >= 1.2
    if np.sum(anclas) < 6:
        anclas = duraciones >= 0.65
    if np.sum(anclas) < 4:
        anclas = np.ones(n, dtype=bool)
    X_anc = X[anclas]

    if num_hablantes <= 0:
        mejor_k = 2 if len(X_anc) >= 4 else 1
        mejor_score = -1.0
        mejor_labels_anc = np.ones(len(X_anc), dtype=int)
        Z = linkage(X_anc, method="ward")
        min_tam = max(2, int(len(X_anc) * 0.03))
        for k in range(2, min(31, len(X_anc))):
            lbl = fcluster(Z, t=k, criterion="maxclust")
            conteos = [int(np.sum(lbl == c)) for c in np.unique(lbl)]
            if len(X_anc) >= 20 and min(conteos) < min_tam:
                continue
            sc = _silhouette_score_coseno(X_anc, lbl)
            penalizado = sc - 0.006 * (k - 2)
            if penalizado > mejor_score:
                mejor_score = penalizado
                mejor_k = k
                mejor_labels_anc = lbl
        if mejor_score < 0.025 and len(X_anc) >= 2:
            mejor_labels_anc = fcluster(Z, t=2, criterion="maxclust")
            mejor_k = 2
        labels_anc = mejor_labels_anc
        k_final = mejor_k
    else:
        k_final = min(num_hablantes, len(X_anc))
        if k_final <= 1:
            labels_anc = np.ones(len(X_anc), dtype=int)
        else:
            Z = linkage(X_anc, method="ward")
            labels_anc = fcluster(Z, t=k_final, criterion="maxclust")

    unicos = sorted(np.unique(labels_anc))
    centroides = np.vstack([np.mean(X_anc[labels_anc == c], axis=0) for c in unicos])
    # Usar distancia euclidiana coherente con Ward linkage para no partir en dos al hablante principal
    dist = cdist(X, centroides, metric="euclidean")
    dist = np.nan_to_num(dist, nan=10.0)
    asignados = np.argmin(dist, axis=1)

    for i in range(n):
        if duraciones[i] < 0.85:
            if 0 < i < n - 1 and asignados[i - 1] == asignados[i + 1]:
                asignados[i] = asignados[i - 1]

    mapa_orden = {}
    contador = 1
    for idx, s in enumerate(segmentos):
        c_raw = int(asignados[idx])
        if c_raw not in mapa_orden:
            mapa_orden[c_raw] = f"SPK_{contador}"
            contador += 1
        s["id"] = idx
        s["hablante"] = mapa_orden[c_raw]
        s["confianza"] = 0.85
        s.setdefault("silenciado", False)

    spk_unicos = sorted(list(set(s["hablante"] for s in segmentos)))
    centroides_dict = {}
    for spk_id in spk_unicos:
        idx_spk = [i for i, s in enumerate(segmentos) if s["hablante"] == spk_id]
        if idx_spk:
            centroides_dict[spk_id] = np.mean(X[idx_spk], axis=0)

    if len(spk_unicos) > 1 and centroides_dict:
        centroides_spks = np.vstack([centroides_dict[spk] for spk in spk_unicos])
        coincidencias_voces = emparejar_con_voces_registradas(centroides_spks, spk_unicos)
    else:
        coincidencias_voces = {}

    nombres_auto = {}
    for spk_id, match in coincidencias_voces.items():
        nombres_auto[spk_id] = match["nombre"]

    hablantes = _resumen_hablantes(segmentos, nombres_previos=nombres_auto, embeddings_spk=centroides_dict, matches_registro=coincidencias_voces)
    return segmentos, hablantes


def _resumen_hablantes(segmentos: list[dict], nombres_previos: dict | None = None,
                       silenciados_previos: dict | None = None,
                       embeddings_spk: dict | None = None,
                       matches_registro: dict | None = None) -> list[dict]:
    nombres_previos = nombres_previos or {}
    silenciados_previos = silenciados_previos or {}
    embeddings_spk = embeddings_spk or {}
    matches_registro = matches_registro or {}
    grupos = {}
    for s in segmentos:
        spk = s["hablante"]
        grupos.setdefault(spk, []).append(s)

    colores = ["#2563eb", "#059669", "#d97706", "#7c3aed", "#db2777", "#0891b2", "#dc2626", "#4f46e5"]
    hablantes = []
    for idx, (spk, segs) in enumerate(grupos.items()):
        num = spk.split("_")[-1]
        dur_total = sum(max(0.0, float(x["fin"]) - float(x["inicio"])) for x in segs)
        ordenados = sorted(
            segs,
            key=lambda x: (
                0 if 2.2 <= (float(x["fin"]) - float(x["inicio"])) <= 10.0 else 1,
                -float(x.get("confianza", 0.5)),
                -min(float(x["fin"]) - float(x["inicio"]), 10.0)
            )
        )
        muestras = [
            {"inicio": float(m["inicio"]), "fin": float(m["fin"]), "texto": m["texto"][:90]}
            for m in ordenados[:5]
        ]
        match_info = matches_registro.get(spk)
        hablantes.append({
            "id": spk,
            "nombre": nombres_previos.get(spk, f"Hablante {num}"),
            "color": (match_info.get("color") if match_info and match_info.get("color") else colores[idx % len(colores)]),
            "silenciado": bool(silenciados_previos.get(spk, False)),
            "cantidad": len(segs),
            "duracion": round(dur_total, 1),
            "muestras": muestras,
            "auto_etiquetado": bool(match_info and match_info.get("auto_etiquetado")),
            "distancia_registro": match_info.get("distancia") if match_info else None
        })
    return hablantes


# ==============================================================================
# TRANSCRIPCIÓN CON WHISPER (SOPORTA FASTER-WHISPER Y OPENAI-WHISPER CON PROGRESO)
# ==============================================================================

def transcribir_con_progreso(ruta_wav16k: str, modelo: str = "small", idioma: str = "es",
                             cb_progreso=None) -> list[dict]:
    dur = max(0.1, duracion_wav(ruta_wav16k))
    lang = None if idioma == "auto" else idioma

    try:
        from faster_whisper import WhisperModel
        with WHISPER_LOCK:
            clave = ("faster", modelo)
            if clave not in WHISPER_CACHE:
                if cb_progreso:
                    cb_progreso(12, f"Cargando modelo Faster-Whisper ({modelo})…")
                WHISPER_CACHE[clave] = WhisperModel(modelo, device="auto", compute_type="default")
            fw_model = WHISPER_CACHE[clave]

        if cb_progreso:
            cb_progreso(18, "Transcribiendo audio…")
        segments_gen, _info = fw_model.transcribe(ruta_wav16k, language=lang, vad_filter=True, beam_size=5)
        salida = []
        for s in segments_gen:
            txt = s.text.strip()
            if txt:
                salida.append({"inicio": round(float(s.start), 2), "fin": round(float(s.end), 2), "texto": txt})
            if cb_progreso:
                pct = min(84, 18 + int((float(s.end) / dur) * 66))
                cb_progreso(pct, f"Transcribiendo… {int(min(100, (float(s.end) / dur) * 100))}% ({len(salida)} frases)")
        return salida
    except ImportError:
        pass

    import whisper
    import whisper.transcribe
    w_transcribe_mod = sys.modules["whisper.transcribe"]

    with WHISPER_LOCK:
        clave = ("openai", modelo)
        if clave not in WHISPER_CACHE:
            if cb_progreso:
                cb_progreso(12, f"Cargando modelo Whisper '{modelo}'…")
            WHISPER_CACHE[clave] = whisper.load_model(modelo)
        ow_model = WHISPER_CACHE[clave]

    tqdm_original = w_transcribe_mod.tqdm.tqdm

    class TqdmProgreso(tqdm_original):
        def update(self, n=1):
            res = super().update(n)
            if self.total and cb_progreso:
                frac = min(1.0, max(0.0, float(self.n) / float(self.total)))
                pct = 18 + int(frac * 66)
                cb_progreso(pct, f"Transcribiendo con Whisper ({modelo})… {int(frac * 100)}%")
            return res

    if cb_progreso:
        cb_progreso(18, f"Transcribiendo con Whisper ({modelo})… 0%")

    w_transcribe_mod.tqdm.tqdm = TqdmProgreso
    try:
        res = ow_model.transcribe(ruta_wav16k, language=lang, verbose=False, fp16=False)
    finally:
        w_transcribe_mod.tqdm.tqdm = tqdm_original

    salida = []
    for s in res.get("segments", []):
        txt = (s.get("text") or "").strip()
        if txt:
            salida.append({
                "inicio": round(float(s["start"]), 2),
                "fin": round(float(s["end"]), 2),
                "texto": txt
            })
    return salida


# ==============================================================================
# AISLAR VOZ (ENTRENAMIENTO DE MODELO), SILENCIAR O RECORTAR HABLANTES Y RANGOS
# ==============================================================================

def _unir_intervalos(intervalos: list[tuple[float, float]], margen_union: float = 0.25) -> list[tuple[float, float]]:
    if not intervalos:
        return []
    ordenados = sorted((max(0.0, float(a)), max(0.0, float(b))) for a, b in intervalos if b > a)
    if not ordenados:
        return []
    unidos = [ordenados[0]]
    for a, b in ordenados[1:]:
        u_a, u_b = unidos[-1]
        if a <= u_b + margen_union:
            unidos[-1] = (u_a, max(u_b, b))
        else:
            unidos.append((a, b))
    return unidos


def _decodificar_pcm(ruta_entrada: str, sr: int = 44100, ch: int = 1) -> np.ndarray:
    cmd_dec = [
        "ffmpeg", "-y", "-i", ruta_entrada,
        "-vn", "-ac", str(ch), "-ar", str(sr), "-f", "s16le", "pipe:1"
    ]
    p_dec = subprocess.run(cmd_dec, capture_output=True)
    if p_dec.returncode != 0:
        raise RuntimeError("Error al leer el audio base con ffmpeg.")
    muestras = np.frombuffer(p_dec.stdout, dtype=np.int16).copy()
    n_frames = len(muestras) // ch
    return muestras[:n_frames * ch].reshape((n_frames, ch))


def _guardar_audio_salida(pcm: np.ndarray, sr: int, ch: int, ruta_salida: str,
                          formato: str = "mp3", cal: str = "192k") -> None:
    if formato == "wav":
        with wave.open(ruta_salida, "wb") as wf:
            wf.setnchannels(ch)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(pcm.tobytes())
    else:
        cmd_enc = [
            "ffmpeg", "-y",
            "-f", "s16le", "-ar", str(sr), "-ac", str(ch), "-i", "pipe:0",
            "-c:a", "libmp3lame", "-b:a", cal, ruta_salida
        ]
        p_enc = subprocess.run(cmd_enc, input=pcm.tobytes(), capture_output=True)
        if p_enc.returncode != 0:
            raise RuntimeError("Error al codificar el MP3 resultante.")


def generar_audio_procesado(ruta_entrada: str, ruta_salida: str,
                            intervalos_silenciados: list[tuple[float, float]],
                            intervalos_activos: list[dict] | None = None,
                            modo: str = "aislar", formato: str = "mp3",
                            cal: str = "192k", mono: bool = False,
                            max_minutos: float = 0.0, min_dur_seg: float = 0.0,
                            pausa_seg: float = 0.30,
                            rango_inicio: float | None = None,
                            rango_fin: float | None = None) -> dict:
    """Genera un archivo de audio procesado según el modo:
    - 'rango_directo': recorta exactamente entre `rango_inicio` y `rango_fin` (detectados por frase o manual)
      sin filtrar hablantes intermedios.
    - 'aislar': concatena ÚNICAMENTE los fragmentos activos seleccionados (ideal para entrenar modelo
      de voz, descartando silencios largos, ruidos no transcritos y voces ajenas).
    - 'recortar': elimina del audio original los intervalos silenciados.
    - 'silenciar': pone en volumen 0 los intervalos silenciados manteniendo la duración original.
    """
    sr = 44100
    ch = 1 if mono else 2
    muestras = _decodificar_pcm(ruta_entrada, sr=sr, ch=ch)
    n_frames = len(muestras)
    dur_original = n_frames / float(sr)
    fade_len = int(0.015 * sr)

    t_rango_ini = max(0.0, float(rango_inicio)) if rango_inicio is not None else 0.0
    t_rango_fin = min(dur_original, float(rango_fin)) if rango_fin is not None else dur_original

    if modo == "rango_directo":
        i0 = max(0, min(n_frames, int(round(t_rango_ini * sr))))
        i1 = max(0, min(n_frames, int(round(t_rango_fin * sr))))
        if i1 <= i0:
            raise ValueError("El tiempo de fin debe ser mayor al tiempo de inicio.")
        seg = muestras[i0:i1].astype(np.float32)
        m = min(fade_len, len(seg) // 3)
        if m > 1:
            seg[:m] *= np.linspace(0.0, 1.0, m, dtype=np.float32)[:, None]
            seg[-m:] *= np.linspace(1.0, 0.0, m, dtype=np.float32)[:, None]
        salida_pcm = seg.astype(np.int16)
        n_usados = 1

    elif modo == "aislar":
        activos = []
        for s in (intervalos_activos or []):
            a = max(float(s["inicio"]), t_rango_ini)
            b = min(float(s["fin"]), t_rango_fin)
            if b > a + 0.15:
                copia = dict(s)
                copia["inicio"] = a
                copia["fin"] = b
                activos.append(copia)

        if min_dur_seg > 0:
            activos = [s for s in activos if (float(s["fin"]) - float(s["inicio"])) >= min_dur_seg]

        if not activos:
            raise ValueError("No quedaron fragmentos activos dentro del rango y filtro seleccionados.")

        if max_minutos > 0:
            max_seg = max_minutos * 60.0
            orden_calidad = sorted(
                activos,
                key=lambda s: (
                    0 if 2.0 <= (float(s["fin"]) - float(s["inicio"])) <= 12.0 else 1,
                    -float(s.get("confianza", 0.8)),
                    -(float(s["fin"]) - float(s["inicio"]))
                )
            )
            elegidos = []
            acum_seg = 0.0
            for s in orden_calidad:
                d_s = max(0.0, float(s["fin"]) - float(s["inicio"]))
                if acum_seg + d_s <= max_seg or not elegidos:
                    elegidos.append(s)
                    acum_seg += d_s
                if acum_seg >= max_seg:
                    break
            activos = sorted(elegidos, key=lambda s: float(s["inicio"]))

        pausa_pcm = np.zeros((int(max(0.0, pausa_seg) * sr), ch), dtype=np.int16)
        trozos = []
        for idx, s in enumerate(activos):
            a = max(0.0, float(s["inicio"]))
            b = min(dur_original, float(s["fin"]))
            if (b - a) > 0.6:
                a += 0.04
                b -= 0.04
            i0 = max(0, min(n_frames, int(round(a * sr))))
            i1 = max(0, min(n_frames, int(round(b * sr))))
            if i1 <= i0:
                continue
            seg = muestras[i0:i1].astype(np.float32)
            m = min(fade_len, len(seg) // 3)
            if m > 1:
                seg[:m] *= np.linspace(0.0, 1.0, m, dtype=np.float32)[:, None]
                seg[-m:] *= np.linspace(1.0, 0.0, m, dtype=np.float32)[:, None]
            trozos.append(seg.astype(np.int16))
            if idx < len(activos) - 1 and len(pausa_pcm) > 0:
                trozos.append(pausa_pcm)

        salida_pcm = np.concatenate(trozos, axis=0) if trozos else np.zeros((int(0.5 * sr), ch), dtype=np.int16)
        n_usados = len(activos)

    elif modo == "recortar":
        unidos = _unir_intervalos(intervalos_silenciados, margen_union=0.25)
        conservados = []
        cursor = t_rango_ini
        for a, b in unidos:
            if b <= t_rango_ini or a >= t_rango_fin:
                continue
            a_c = max(t_rango_ini, a)
            b_c = min(t_rango_fin, b)
            if a_c > cursor + 0.04:
                conservados.append((cursor, min(a_c, t_rango_fin)))
            cursor = max(cursor, b_c)
        if cursor < t_rango_fin - 0.04:
            conservados.append((cursor, t_rango_fin))

        trozos = []
        for a, b in conservados:
            i0 = max(0, min(n_frames, int(round(a * sr))))
            i1 = max(0, min(n_frames, int(round(b * sr))))
            if i1 <= i0:
                continue
            seg = muestras[i0:i1].astype(np.float32)
            m = min(fade_len, len(seg) // 3)
            if m > 1:
                seg[:m] *= np.linspace(0.0, 1.0, m, dtype=np.float32)[:, None]
                seg[-m:] *= np.linspace(1.0, 0.0, m, dtype=np.float32)[:, None]
            trozos.append(seg.astype(np.int16))

        salida_pcm = np.concatenate(trozos, axis=0) if trozos else np.zeros((int(0.5 * sr), ch), dtype=np.int16)
        n_usados = len(conservados)

    else:
        unidos = _unir_intervalos(intervalos_silenciados, margen_union=0.25)
        salida_pcm = muestras.copy()
        for a, b in unidos:
            i0 = max(0, min(n_frames, int(round(a * sr))))
            i1 = max(0, min(n_frames, int(round(b * sr))))
            if i1 <= i0:
                continue
            f_out_ini = max(0, i0 - fade_len)
            if i0 > f_out_ini:
                rampa = np.linspace(1.0, 0.0, i0 - f_out_ini, dtype=np.float32)[:, None]
                salida_pcm[f_out_ini:i0] = (salida_pcm[f_out_ini:i0].astype(np.float32) * rampa).astype(np.int16)
            salida_pcm[i0:i1] = 0
            f_in_fin = min(n_frames, i1 + fade_len)
            if f_in_fin > i1:
                rampa = np.linspace(0.0, 1.0, f_in_fin - i1, dtype=np.float32)[:, None]
                salida_pcm[i1:f_in_fin] = (salida_pcm[i1:f_in_fin].astype(np.float32) * rampa).astype(np.int16)
        # Si además se definió un rango [rango_inicio, rango_fin], recortar a ese tramo
        if rango_inicio is not None or rango_fin is not None:
            r0 = max(0, min(n_frames, int(round(t_rango_ini * sr))))
            r1 = max(0, min(n_frames, int(round(t_rango_fin * sr))))
            if r1 > r0:
                salida_pcm = salida_pcm[r0:r1]
        n_usados = len(unidos)

    dur_final = len(salida_pcm) / float(sr)
    _guardar_audio_salida(salida_pcm, sr, ch, ruta_salida, formato=formato, cal=cal)

    return {
        "duracion_original": round(dur_original, 2),
        "duracion_final": round(dur_final, 2),
        "fragmentos_incluidos": n_usados,
    }


def generar_zip_dataset_voz(ruta_entrada: str, ruta_zip: str,
                            intervalos_activos: list[dict],
                            max_minutos: float = 0.0, min_dur_seg: float = 1.0,
                            rango_inicio: float | None = None,
                            rango_fin: float | None = None) -> dict:
    sr = 44100
    muestras = _decodificar_pcm(ruta_entrada, sr=sr, ch=1)
    n_frames = len(muestras)
    dur_original = n_frames / float(sr)
    fade_len = int(0.012 * sr)

    t_rango_ini = max(0.0, float(rango_inicio)) if rango_inicio is not None else 0.0
    t_rango_fin = min(dur_original, float(rango_fin)) if rango_fin is not None else dur_original

    activos = []
    for s in intervalos_activos:
        a = max(float(s["inicio"]), t_rango_ini)
        b = min(float(s["fin"]), t_rango_fin)
        if (b - a) >= max(0.4, min_dur_seg):
            copia = dict(s)
            copia["inicio"] = a
            copia["fin"] = b
            activos.append(copia)

    if not activos:
        raise ValueError("No hay fragmentos activos que superen la duración mínima dentro del rango.")

    if max_minutos > 0:
        max_seg = max_minutos * 60.0
        elegidos = []
        acum = 0.0
        for s in sorted(activos, key=lambda x: -float(x.get("confianza", 0.8))):
            d = float(s["fin"]) - float(s["inicio"])
            if acum + d <= max_seg or not elegidos:
                elegidos.append(s)
                acum += d
            if acum >= max_seg:
                break
        activos = sorted(elegidos, key=lambda x: float(x["inicio"]))

    lineas_csv = []
    dur_total = 0.0
    with zipfile.ZipFile(ruta_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for idx, s in enumerate(activos, start=1):
            a = max(0.0, float(s["inicio"]))
            b = min(dur_original, float(s["fin"]))
            i0 = max(0, min(n_frames, int(round(a * sr))))
            i1 = max(0, min(n_frames, int(round(b * sr))))
            if i1 <= i0:
                continue
            seg = muestras[i0:i1].astype(np.float32)
            m = min(fade_len, len(seg) // 3)
            if m > 1:
                seg[:m] *= np.linspace(0.0, 1.0, m, dtype=np.float32)[:, None]
                seg[-m:] *= np.linspace(1.0, 0.0, m, dtype=np.float32)[:, None]
            pcm_clip = seg.astype(np.int16)
            dur_total += len(pcm_clip) / float(sr)

            nombre_wav = f"wavs/clip_{idx:04d}.wav"
            buf = io.BytesIO()
            with wave.open(buf, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sr)
                wf.writeframes(pcm_clip.tobytes())
            zf.writestr(nombre_wav, buf.getvalue())
            texto_limpio = (s.get("texto") or "").replace("|", " ").replace("\n", " ").strip()
            lineas_csv.append(f"clip_{idx:04d}.wav|{texto_limpio}")

        zf.writestr("metadata.csv", "\n".join(lineas_csv).encode("utf-8"))

    return {
        "duracion_final": round(dur_total, 2),
        "fragmentos_incluidos": len(lineas_csv),
    }


# ==============================================================================
# TRABAJOS EN SEGUNDO PLANO (EXTRACCIÓN + TRANSCRIPCIÓN + HABLANTES + FRASES)
# ==============================================================================

def _actualizar_job(job_id: str, **cambios) -> None:
    with JOBS_LOCK:
        if job_id in JOBS:
            JOBS[job_id].update(cambios)


def _worker_analizar(job_id: str, ruta_entrada: str, es_temporal: bool,
                     nombre_original: str, params: dict) -> None:
    try:
        cal = params.get("cal", "192k")
        if cal not in ("64k", "128k", "192k", "320k"):
            cal = "192k"
        desde = params.get("desde", "").strip()
        hasta = params.get("hasta", "").strip()
        frase_ini = params.get("frase_ini", "").strip()
        frase_fin = params.get("frase_fin", "").strip()
        mono = bool(params.get("mono", False))
        mb = float(params.get("mb", 0) or 0)
        solo_extraer = bool(params.get("solo_extraer", False))
        modo_p1 = params.get("modo_p1", "mp3")
        modelo = params.get("modelo", "small")
        if modelo not in ("tiny", "base", "small", "medium"):
            modelo = "small"
        idioma = params.get("idioma", "es")
        num_hablantes = int(params.get("hablantes", 0) or 0)
        usar_mp3_existente = bool(params.get("usar_mp3_existente", False))

        base = os.path.splitext(nombre_original)[0]
        suf_tramo = ""
        if desde or hasta:
            d_lbl = (desde or "00-00").replace(":", "-")
            h_lbl = (hasta or "fin").replace(":", "-")
            suf_tramo = f"_recorte_{d_lbl}_a_{h_lbl}"

        nombre_video_out = ""
        nombre_mp3 = ""
        ruta_mp3 = ""

        if solo_extraer and params.get("split_modo") in ("puntos", "duracion"):
            _worker_dividir(job_id, ruta_entrada, base, params)
            return

        if solo_extraer and modo_p1 in ("video_mp4", "ambos"):
            _actualizar_job(job_id, progreso=15, mensaje="Recortando video MP4 en el minuto y segundo indicados…")
            nombre_video_out = nombre_unico(SALIDA, base + (suf_tramo or "_recortado"), ".mp4")
            ruta_video_out = os.path.join(SALIDA, nombre_video_out)
            recortar_video_ffmpeg(ruta_entrada, ruta_video_out, desde=desde, hasta=hasta)
            if modo_p1 == "video_mp4":
                _actualizar_job(
                    job_id,
                    estado="completado",
                    progreso=100,
                    mensaje="Video recortado correctamente.",
                    resultado={"modo": "solo_video", "nombre_video": nombre_video_out, "nombre_mp3": ""}
                )
                return

        if usar_mp3_existente and not desde and not hasta and mb <= 0:
            nombre_mp3 = os.path.basename(ruta_entrada)
            ruta_mp3 = ruta_entrada
        else:
            _actualizar_job(job_id, progreso=45 if nombre_video_out else 5, mensaje="Extrayendo / recortando audio MP3 con ffmpeg…")
            nombre_mp3 = nombre_unico(SALIDA, base + suf_tramo, ".mp3")
            ruta_mp3 = os.path.join(SALIDA, nombre_mp3)
            extraer_mp3_ffmpeg(ruta_entrada, ruta_mp3, cal=cal, desde=desde, hasta=hasta, mb=mb, mono=mono)

        if solo_extraer:
            _actualizar_job(
                job_id,
                estado="completado",
                progreso=100,
                mensaje="Procesamiento completado.",
                resultado={"modo": "solo_mp3", "nombre_mp3": nombre_mp3, "nombre_video": nombre_video_out}
            )
            return

        sesion_id = uuid.uuid4().hex[:12]
        ruta_wav16k = os.path.join(TEMP_DIR, f"{sesion_id}_16k.wav")
        _actualizar_job(job_id, progreso=10, mensaje="Preparando audio a 16 kHz para transcripción…")
        convertir_a_wav16k(ruta_mp3, ruta_wav16k)
        duracion = duracion_wav(ruta_wav16k)

        def cb_prog(pct, msg):
            _actualizar_job(job_id, progreso=pct, mensaje=msg)

        segmentos_raw = transcribir_con_progreso(ruta_wav16k, modelo=modelo, idioma=idioma, cb_progreso=cb_prog)

        _actualizar_job(job_id, progreso=88, mensaje="Analizando voces y agrupando hablantes…")
        wav_data = cargar_audio_16k(ruta_wav16k)
        segmentos_div = subdividir_segmentos_largos(segmentos_raw, wav_data, 16000)
        X = calcular_matriz_features(segmentos_div, wav_data, 16000)
        segmentos, hablantes = agrupar_hablantes(segmentos_div, X, num_hablantes=num_hablantes)

        # Si el usuario ya ingresó una frase de inicio y/o frase de fin en el Paso 2, buscar sus marcas de tiempo
        rango_detectado = {
            "frase_ini": frase_ini,
            "frase_fin": frase_fin,
            "inicio_seg": None,
            "fin_seg": None,
            "match_ini": None,
            "match_fin": None,
        }
        if frase_ini:
            m_ini = buscar_frases_en_segmentos(segmentos, frase_ini, desde_seg=0.0)
            if m_ini:
                rango_detectado["inicio_seg"] = m_ini[0]["frase_inicio"]
                rango_detectado["match_ini"] = m_ini[0]
        if frase_fin:
            t_min = rango_detectado["inicio_seg"] or 0.0
            m_fin = buscar_frases_en_segmentos(segmentos, frase_fin, desde_seg=t_min)
            if m_fin:
                rango_detectado["fin_seg"] = m_fin[0]["frase_fin"]
                rango_detectado["match_fin"] = m_fin[0]

        sesion_data = {
            "sesion_id": sesion_id,
            "nombre_original": nombre_original,
            "nombre_base": re.sub(r'[<>:"/\\|?*]', "_", base).strip() or "audio",
            "nombre_mp3": nombre_mp3,
            "ruta_mp3": ruta_mp3,
            "ruta_wav16k": ruta_wav16k,
            "cal": cal,
            "mono": mono,
            "duracion": round(duracion, 2),
            "modelo": modelo,
            "segmentos": segmentos,
            "hablantes": hablantes,
            "X": X,
            "offset": 0.0,
            "sincronizado": True,
            "origen_media": "whisper",
        }
        with JOBS_LOCK:
            SESIONES[sesion_id] = sesion_data
        guardar_sesion_disco(sesion_data)

        _actualizar_job(
            job_id,
            estado="completado",
            progreso=100,
            mensaje="Transcripción y detección de hablantes completada.",
            resultado={
                "modo": "analisis",
                "sesion_id": sesion_id,
                "offset": 0.0,
                "sincronizado": True,
                "origen_media": "whisper",
                "nombre_original": nombre_original,
                "nombre_mp3": nombre_mp3,
                "duracion": round(duracion, 2),
                "modelo": modelo,
                "hablantes": hablantes,
                "segmentos": segmentos,
                "rango_detectado": rango_detectado,
            }
        )
    except Exception as e:
        _actualizar_job(job_id, estado="error", mensaje=str(e), error=str(e))
    finally:
        if es_temporal and os.path.exists(ruta_entrada):
            try:
                os.remove(ruta_entrada)
            except OSError:
                pass


# ==============================================================================
# INTERFAZ WEB (HTML + CSS + JS)
# ==============================================================================


import csv
import xml.etree.ElementTree as ET

SESIONES_DIR = os.path.join(BASE_DIR, "transcripciones")
os.makedirs(SESIONES_DIR, exist_ok=True)



# ==============================================================================
# MOTOR DE MACHINE LEARNING: ENTRENAR ESTILO DE HABLA Y CONVERTIR GUIONES
# ==============================================================================

ESTILO_MODELO_FILE = os.path.join(BASE_DIR, "estilo_habla_modelo.json")
ESTILO_LOCK = threading.Lock()

STOPWORDS_ES = {
    "el", "la", "los", "las", "un", "una", "unos", "unas", "de", "del", "al", "a",
    "en", "con", "por", "para", "sin", "sobre", "entre", "hasta", "desde", "que",
    "y", "e", "o", "u", "pero", "sino", "como", "cuando", "donde", "quien", "cual",
    "se", "me", "te", "le", "nos", "les", "lo", "su", "sus", "mi", "mis", "tu", "tus",
    "es", "son", "fue", "era", "ser", "esta", "estan", "este", "esta", "estos", "estas",
    "eso", "esa", "ese", "esos", "esas", "aquel", "todo", "toda", "todos", "todas",
    "mas", "menos", "muy", "ya", "no", "si", "tambien", "solo", "porque", "pues",
    "hay", "tiene", "tienen", "hacer", "hace", "puede", "pueden", "ir", "va", "van"
}

PATRONES_MARCADORES_HABLA = [
    "vamos a ver", "vamos a hacer", "vamos a poner", "vamos a crear", "vamos a usar",
    "lo que vamos a", "entonces vamos a", "yo les voy a", "les voy a", "lo voy a",
    "voy a crear", "voy a hacer", "voy a poner", "van a tener", "van a ver",
    "o sea", "por ejemplo", "digamos", "básicamente", "basicamente", "entonces",
    "fíjense que", "fijense que", "fíjense", "fijate que", "fijate", "miren que",
    "vieron que", "acá tenemos", "aca tenemos", "acá", "aca", "un montón de",
    "un monton de", "no sé si", "no se si", "es lo mismo", "dentro de la",
    "a través de", "a traves de", "en este caso", "por otro lado", "de alguna manera",
    "qué pasa si", "que pasa si", "se entiende", "me explico", "está bien", "esta bien",
    "bueno", "claro", "exacto", "joya", "genial", "perfecto", "cuestión que"
]

REEMPLAZOS_BASE_ESTILO = [
    (r"\ba continuación veremos\b", "ahora vamos a ver"),
    (r"\ba continuación\b", "ahora"),
    (r"\bes importante destacar que\b", "fíjense que"),
    (r"\bes importante mencionar que\b", "tengan en cuenta que"),
    (r"\bcabe destacar que\b", "fíjense que"),
    (r"\bdebemos considerar\b", "tenemos que tener en cuenta"),
    (r"\bdebemos\b", "tenemos que"),
    (r"\bdebes\b", "tenés que"),
    (r"\bpuedes\b", "podés"),
    (r"\btienes\b", "tenés"),
    (r"\bquieres\b", "querés"),
    (r"\bsabes\b", "sabés"),
    (r"\bhaces\b", "hacés"),
    (r"\bpones\b", "ponés"),
    (r"\bves\b", "ves"),
    (r"\bobserva\b", "fijate"),
    (r"\bobserven\b", "fíjense"),
    (r"\bmira\b", "mirá"),
    (r"\bpiensa\b", "pensá"),
    (r"\brecuerda\b", "acordate"),
    (r"\butilizaremos\b", "vamos a usar"),
    (r"\butilizar\b", "usar"),
    (r"\brealizaremos\b", "vamos a hacer"),
    (r"\bcrearemos\b", "vamos a crear"),
    (r"\bveremos\b", "vamos a ver"),
    (r"\banalizaremos\b", "vamos a analizar"),
    (r"\bdefiniremos\b", "vamos a definir"),
    (r"\bimplementaremos\b", "vamos a implementar"),
    (r"\btendremos\b", "vamos a tener"),
    (r"\btendrán\b", "van a tener"),
    (r"\bpodrán\b", "van a poder"),
    (r"\baquí\b", "acá"),
    (r"\bmuchísimos\b", "un montón de"),
    (r"\bmuchas cosas\b", "un montón de cosas"),
    (r"\bpor consiguiente\b", "entonces"),
    (r"\bpor lo tanto\b", "así que"),
    (r"\bsin embargo\b", "pero"),
    (r"\bno obstante\b", "igual"),
    (r"\bes decir\b", "o sea"),
    (r"\ben conclusión\b", "en resumen"),
    (r"\bfinalmente\b", "y por último"),
]


def cargar_modelo_estilo() -> dict:
    with ESTILO_LOCK:
        if not os.path.exists(ESTILO_MODELO_FILE):
            return {"entrenado": False}
        try:
            with open(ESTILO_MODELO_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {"entrenado": False}


def guardar_modelo_estilo(modelo: dict) -> None:
    with ESTILO_LOCK:
        try:
            with open(ESTILO_MODELO_FILE, "w", encoding="utf-8") as f:
                json.dump(modelo, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Error guardando modelo de estilo: {e}")


def _limpiar_oracion_oral(texto: str) -> str:
    t = re.sub(r"\s+", " ", (texto or "").strip())
    # Quitar repeticiones tartamudeadas inmediatas ("sí sí sí" -> "sí", "el el" -> "el")
    t = re.sub(r"\b(\w+)(\s+\1){1,}\b", r"\1", t, flags=re.IGNORECASE)
    return t.strip()


def entrenar_modelo_estilo_ml(textos_oraciones: list[str], duraciones_seg: list[float] | None = None,
                              fuentes: list[str] | None = None) -> dict:
    """Entrena un modelo estilométrico + estadístico N-gramas + índice TF-IDF sobre las transcripciones del usuario."""
    oraciones_limpias = []
    for t in textos_oraciones:
        limpio = _limpiar_oracion_oral(t)
        if len(limpio.split()) >= 3:
            oraciones_limpias.append(limpio)

    if not oraciones_limpias:
        raise ValueError("No hay suficientes frases con contenido para entrenar el modelo.")

    texto_completo = " ".join(oraciones_limpias)
    texto_lower = texto_completo.lower()
    palabras_todas = re.findall(r"[a-záéíóúüñ]+", texto_lower)
    total_palabras = max(1, len(palabras_todas))
    dur_total = sum(duraciones_seg) if duraciones_seg else (total_palabras / 2.3)
    ppm = round((total_palabras / max(10.0, dur_total)) * 60.0, 1)

    longitudes = [len(o.split()) for o in oraciones_limpias]
    long_media = round(float(np.mean(longitudes)), 1)
    long_mediana = round(float(np.median(longitudes)), 1)

    # 1. Conteo de marcadores discursivos y muletillas reales del usuario
    marcadores_conteo = {}
    for m in PATRONES_MARCADORES_HABLA:
        c = len(re.findall(r"\b" + re.escape(m) + r"\b", texto_lower))
        if c > 0:
            # Unificar variantes sin tilde
            clave = (
                m.replace("basicamente", "básicamente")
                 .replace("fijense", "fíjense")
                 .replace("aca", "acá")
                 .replace("monton", "montón")
                 .replace("no se si", "no sé si")
                 .replace("a traves de", "a través de")
                 .replace("que pasa si", "qué pasa si")
                 .replace("esta bien", "está bien")
            )
            marcadores_conteo[clave] = marcadores_conteo.get(clave, 0) + c

    # 2. Descubrir N-gramas (2, 3 y 4 palabras) propios más frecuentes automáticamente
    from collections import Counter
    bigramas = Counter(zip(palabras_todas, palabras_todas[1:]))
    trigramas = Counter(zip(palabras_todas, palabras_todas[1:], palabras_todas[2:]))
    cuatrigramas = Counter(zip(palabras_todas, palabras_todas[1:], palabras_todas[2:], palabras_todas[3:]))

    frases_frecuentes = []
    vistos_frases = set()
    for gram_counter, min_c in ((cuatrigramas, 3), (trigramas, 4), (bigramas, 8)):
        for tup, cnt in gram_counter.most_common(40):
            frase = " ".join(tup)
            if all(w in STOPWORDS_ES for w in tup) and frase not in ("o sea", "lo que", "para que", "es que", "si no"):
                continue
            if frase not in vistos_frases and cnt >= min_c:
                vistos_frases.add(frase)
                frases_frecuentes.append({"frase": frase, "frecuencia": int(cnt)})

    top_marcadores = sorted(
        [{"frase": k, "frecuencia": v} for k, v in marcadores_conteo.items()],
        key=lambda x: -x["frecuencia"]
    )[:25]

    # 3. Extraer aperturas reales de oración (cómo empieza a explicar una idea)
    aperturas_counter = Counter()
    cierres_counter = Counter()
    for o in oraciones_limpias:
        toks = re.findall(r"[a-záéíóúüñ]+", o.lower())
        if len(toks) >= 6:
            ap2 = " ".join(toks[:2])
            ap3 = " ".join(toks[:3])
            if ap3 in ("vamos a ver", "vamos a hacer", "lo que vamos", "entonces vamos a", "por otro lado", "en este caso", "fijense que aca", "fíjense que acá", "la idea es", "aca lo que", "acá lo que"):
                aperturas_counter[ap3.capitalize()] += 1
            elif ap2 in ("entonces", "bueno", "ahora", "fíjense", "fijense", "digamos", "básicamente", "basicamente", "acá", "aca", "por ejemplo", "o sea", "vean que"):
                aperturas_counter[ap2.capitalize()] += 1
            ci2 = " ".join(toks[-2:])
            if ci2 in ("se entiende", "está bien", "esta bien", "o no", "me explico", "digamos", "por ejemplo", "básicamente", "es lo mismo"):
                cierres_counter[ci2] += 1

    aperturas_tipicas = [k for k, _ in aperturas_counter.most_common(12)]
    if not aperturas_tipicas:
        aperturas_tipicas = ["Bueno, entonces", "Ahora vamos a ver", "Fíjense que", "Básicamente", "En este caso"]

    cierres_tipicos = [k for k, _ in cierres_counter.most_common(8)]
    if not cierres_tipicos:
        cierres_tipicos = ["¿se entiende?", "¿está bien?", "digamos", "por ejemplo"]

    # 4. Vocabulario distintivo del hablante (TF alto excluyendo stopwords comunes)
    vocab_counter = Counter(w for w in palabras_todas if len(w) >= 4 and w not in STOPWORDS_ES)
    vocab_frecuente = [{"palabra": w, "frecuencia": int(c)} for w, c in vocab_counter.most_common(40)]

    # 5. Construir bloques explicativos reales (de 25 a 65 palabras) para RAG (Retrieval-Augmented Style Transfer)
    bloques_rag = []
    buf = []
    buf_words = 0
    for o in oraciones_limpias:
        n_w = len(o.split())
        buf.append(o)
        buf_words += n_w
        if buf_words >= 32:
            bloque_txt = " ".join(buf)
            kws = list(set(w for w in re.findall(r"[a-záéíóúüñ]{4,}", bloque_txt.lower()) if w not in STOPWORDS_ES))
            bloques_rag.append({"texto": bloque_txt, "keywords": kws[:25]})
            buf = []
            buf_words = 0
    if buf and buf_words >= 12:
        bloque_txt = " ".join(buf)
        kws = list(set(w for w in re.findall(r"[a-záéíóúüñ]{4,}", bloque_txt.lower()) if w not in STOPWORDS_ES))
        bloques_rag.append({"texto": bloque_txt, "keywords": kws[:25]})

    # Conservar hasta 250 bloques representativos
    bloques_rag = bloques_rag[:250]

    # Preservar muletillas personalizadas si ya existían
    previo = cargar_modelo_estilo()
    muletillas_custom = previo.get("muletillas_personalizadas", [])
    if not muletillas_custom:
        muletillas_custom = [m["frase"] for m in top_marcadores[:10]] or ["vamos a ver", "o sea", "por ejemplo", "entonces", "acá", "un montón de"]

    modelo = {
        "entrenado": True,
        "fecha_entrenamiento": time.strftime("%Y-%m-%d %H:%M:%S"),
        "fuentes": fuentes or ["Transcripción de sesión"],
        "metricas": {
            "total_intervenciones": len(oraciones_limpias),
            "total_palabras": total_palabras,
            "palabras_por_oracion_media": long_media,
            "palabras_por_oracion_mediana": long_mediana,
            "palabras_por_minuto": ppm,
            "total_bloques_rag": len(bloques_rag),
        },
        "top_marcadores": top_marcadores,
        "frases_frecuentes": frases_frecuentes[:25],
        "aperturas_tipicas": aperturas_tipicas,
        "cierres_tipicos": cierres_tipicos,
        "vocabulario_frecuente": vocab_frecuente,
        "muletillas_personalizadas": muletillas_custom,
        "bloques_rag": bloques_rag,
    }
    guardar_modelo_estilo(modelo)
    return modelo


def recuperar_ejemplos_similares_rag(texto_consulta: str, modelo: dict, top_k: int = 4) -> list[str]:
    """Busca en el corpus entrenado del usuario los fragmentos reales más parecidos al guión actual."""
    bloques = modelo.get("bloques_rag", [])
    if not bloques:
        return []
    kws_q = set(w for w in re.findall(r"[a-záéíóúüñ]{4,}", (texto_consulta or "").lower()) if w not in STOPWORDS_ES)
    puntuados = []
    for idx, b in enumerate(bloques):
        kws_b = set(b.get("keywords", []))
        inter = len(kws_q & kws_b)
        # Bonus si el bloque tiene marcadores conversacionales ricos
        txt_b = b.get("texto", "")
        bonus = sum(0.35 for m in modelo.get("muletillas_personalizadas", [])[:8] if m.lower() in txt_b.lower())
        score = inter * 2.0 + bonus
        puntuados.append((score, idx, txt_b))
    puntuados.sort(key=lambda x: (-x[0], x[1]))
    return [x[2] for x in puntuados[:top_k]]


def _adaptar_oracion_a_estilo(oracion: str, idx_oracion: int, total_oraciones: int,
                              intensidad: str, muletillas: list[str], aperturas: list[str],
                              cierres: list[str]) -> str:
    txt = oracion.strip()
    if not txt:
        return ""

    # 1. Aplicar transformaciones léxicas (formal/neutro -> voseo y expresiones propias del usuario)
    for patron, reemplazo in REEMPLAZOS_BASE_ESTILO:
        def _repl(match):
            orig = match.group(0)
            if orig[0].isupper():
                return reemplazo[0].upper() + reemplazo[1:]
            return reemplazo
        txt = re.sub(patron, _repl, txt, flags=re.IGNORECASE)

    # 2. Si la oración es excesivamente larga y formal (con ";" o ":"), dividirla al ritmo oral del usuario
    if ";" in txt:
        partes_pc = [p.strip() for p in txt.split(";") if p.strip()]
        if len(partes_pc) > 1:
            conector = "o sea, " if "o sea" in [m.lower() for m in muletillas] else "entonces, "
            txt = ". ".join(p[0].upper() + p[1:] if p else "" for p in partes_pc)

    # 3. Inyectar aperturas y conectores reales aprendidos según intensidad
    ya_empieza_oral = any(
        txt.lower().startswith(pref)
        for pref in ("bueno", "entonces", "ahora", "fíjense", "fijate", "mirá", "o sea", "acá", "básicamente", "vamos a")
    )

    if not ya_empieza_oral and len(txt.split()) >= 5:
        if idx_oracion == 0 and intensidad in ("equilibrado", "conversacional"):
            ap = "Bueno, hoy vamos a ver " if txt.lower().startswith(("el ", "la ", "los ", "las ")) and intensidad == "conversacional" else "Bueno, entonces "
            if ap.endswith("ver "):
                txt = ap + txt[0].lower() + txt[1:]
            else:
                txt = ap + txt[0].lower() + txt[1:]
        elif idx_oracion > 0:
            if intensidad == "conversacional" and idx_oracion % 2 == 1:
                conectores = [
                    "Entonces, ",
                    "Fíjense que ",
                    "Y acá lo que vamos a ver es que ",
                    "O sea, ",
                    "Básicamente, ",
                ]
                elegido = conectores[idx_oracion % len(conectores)]
                txt = elegido + txt[0].lower() + txt[1:]
            elif intensidad == "equilibrado" and idx_oracion % 3 == 1:
                conectores_eq = ["Entonces, ", "Fíjense que ", "O sea, ", "Además, acá "]
                elegido = conectores_eq[idx_oracion % len(conectores_eq)]
                txt = elegido + txt[0].lower() + txt[1:]

    # 4. Si en el medio hay comas explicativas y la intensidad es conversacional/equilibrada, usar marcadores propios
    if intensidad in ("equilibrado", "conversacional") and "por ejemplo" not in txt.lower():
        txt = re.sub(r",\s*como puede ser\b", ", por ejemplo,", txt, flags=re.IGNORECASE)
        txt = re.sub(r"\btales como\b", "como por ejemplo", txt, flags=re.IGNORECASE)

    # 5. Cierre de bloque en modo conversacional
    if intensidad == "conversacional" and idx_oracion == total_oraciones - 1 and not txt.endswith("?"):
        if not txt.endswith("."):
            txt += "."
        cierre_add = " ¿Se entiende la idea?" if "se entiende" in " ".join(cierres).lower() else " ¿Vamos bien hasta acá?"
        txt = txt + cierre_add

    return txt


def convertir_guion_a_mi_estilo(guion_original: str, intensidad: str = "equilibrado",
                                formato_salida: str = "fluido",
                                usar_ollama: bool = False, modelo_ollama: str = "llama3") -> dict:
    """Convierte un guión o texto base al estilo de habla aprendido en el modelo ML."""
    guion_original = (guion_original or "").strip()
    if not guion_original:
        raise ValueError("Escribí o pegá un guión para convertirlo a tu estilo.")

    modelo = cargar_modelo_estilo()
    if not modelo.get("entrenado"):
        # Intentar auto-entrenar con las sesiones disponibles si aún no se entrenó
        textos_auto = []
        durs_auto = []
        fuentes_auto = []
        with JOBS_LOCK:
            for sid, ses in SESIONES.items():
                segs = ses.get("segmentos", [])
                if segs:
                    # Usar el hablante principal (SPK_1) o todos los activos
                    spk1 = [s for s in segs if s.get("hablante") == "SPK_1"] or segs
                    for s in spk1:
                        textos_auto.append(s.get("texto", ""))
                        durs_auto.append(max(0.5, float(s.get("fin", 0)) - float(s.get("inicio", 0))))
                    fuentes_auto.append(ses.get("nombre_original", sid))
        if textos_auto:
            modelo = entrenar_modelo_estilo_ml(textos_auto, durs_auto, fuentes_auto)

    muletillas = modelo.get("muletillas_personalizadas") or ["vamos a ver", "o sea", "por ejemplo", "entonces", "acá"]
    aperturas = modelo.get("aperturas_tipicas") or ["Bueno, entonces", "Vamos a ver"]
    cierres = modelo.get("cierres_tipicos") or ["¿se entiende?"]
    ejemplos_rag = recuperar_ejemplos_similares_rag(guion_original, modelo, top_k=4)

    # Construir el Prompt Maestro con el ADN de estilo + RAG (útil tanto para Ollama como para copiar)
    top_m_str = ", ".join(f'"{m}"' for m in muletillas[:12])
    ejemplos_str = "\n".join(f'- "{ej}"' for ej in ejemplos_rag[:4])
    prompt_adn = (
        "Sos un clon exacto de mi forma de hablar y expresarme en español rioplatense (Argentina).\n"
        f"Mi ritmo promedio es de {modelo.get('metricas', {}).get('palabras_por_oracion_media', 15)} palabras por oración "
        f"({modelo.get('metricas', {}).get('palabras_por_minuto', 135)} palabras por minuto).\n"
        f"Expresiones y conectores reales que más uso al hablar: {top_m_str}.\n"
        "Uso voseo natural (tenés, fijate, mirá, vamos a ver, vamos a crear, les voy a mostrar, acá, un montón de).\n\n"
        "Ejemplos reales extraídos de mis grabaciones (imitá este tono, ritmo y estructura sintáctica):\n"
        f"{ejemplos_str}\n\n"
        "TAREA: Reescribí el siguiente guión para que suene 100% dicho por mí de forma oral y natural, "
        "manteniendo todos los conceptos técnicos pero hablándolo exactamente con mi estilo:\n\n"
        f"GUIÓN ORIGINAL:\n{guion_original}"
    )

    # Si el usuario activó Ollama local, intentar generar con el modelo local primero
    usado_motor = "Motor ML Estilométrico Local + RAG"
    if usar_ollama:
        try:
            req_data = json.dumps({
                "model": modelo_ollama or "llama3",
                "prompt": prompt_adn,
                "stream": False
            }).encode("utf-8")
            req = urllib.request.Request(
                "http://127.0.0.1:11434/api/generate",
                data=req_data,
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=25) as resp:
                r_json = json.loads(resp.read().decode("utf-8"))
                resp_txt = (r_json.get("response") or "").strip()
                if resp_txt:
                    return {
                        "guion_convertido": resp_txt,
                        "ejemplos_referencia": ejemplos_rag,
                        "prompt_adn": prompt_adn,
                        "motor": f"Ollama Local ({modelo_ollama}) + RAG de tus transcripciones",
                        "palabras_salida": len(resp_txt.split()),
                        "duracion_estimada_seg": round(len(resp_txt.split()) / max(1.5, (modelo.get("metricas", {}).get("palabras_por_minuto", 135) / 60.0)), 1)
                    }
        except Exception:
            usado_motor = "Motor ML Estilométrico Local + RAG (Ollama no detectado en puerto 11434, se usó el motor local directo)"

    # Motor ML Estilométrico Local: procesa párrafo por párrafo y oración por oración
    parrafos = [p.strip() for p in re.split(r"\n\s*\n|\n", guion_original) if p.strip()]
    parrafos_convertidos = []
    idx_global = 0
    total_oraciones_est = sum(len(re.split(r"(?<=[.?!])\s+", p)) for p in parrafos)

    for p_idx, parrafo in enumerate(parrafos):
        oraciones = [o.strip() for o in re.split(r"(?<=[.?!])\s+", parrafo) if o.strip()]
        nuevas_oraciones = []
        for o in oraciones:
            adaptada = _adaptar_oracion_a_estilo(
                o, idx_global, max(1, total_oraciones_est),
                intensidad=intensidad,
                muletillas=muletillas,
                aperturas=aperturas,
                cierres=cierres
            )
            if adaptada:
                nuevas_oraciones.append(adaptada)
                idx_global += 1

        if formato_salida == "teleprompter":
            parrafos_convertidos.extend(nuevas_oraciones)
        else:
            parrafos_convertidos.append(" ".join(nuevas_oraciones))

    sep_out = "\n\n• " if formato_salida == "teleprompter" else "\n\n"
    texto_final = ("• " + sep_out.join(parrafos_convertidos)) if formato_salida == "teleprompter" else sep_out.join(parrafos_convertidos)
    n_pal_out = len(texto_final.split())
    ppm_usr = float(modelo.get("metricas", {}).get("palabras_por_minuto", 135) or 135)
    dur_est = round((n_pal_out / max(60.0, ppm_usr)) * 60.0, 1)

    return {
        "guion_convertido": texto_final,
        "ejemplos_referencia": ejemplos_rag,
        "prompt_adn": prompt_adn,
        "motor": usado_motor,
        "palabras_salida": n_pal_out,
        "duracion_estimada_seg": dur_est
    }


def _neutralizar_para_dataset(texto_oral: str) -> str:
    """Genera una versión neutra/formal sintética de un fragmento oral real para armar pares (Guión neutro -> Mi Estilo) de Fine-Tuning."""
    t = texto_oral
    inversos = [
        (r"\bbueno,?\s+entonces\b", ""),
        (r"\bo sea,?\s*", "es decir, "),
        (r"\bfíjense que\b", "es importante observar que"),
        (r"\bfijense que\b", "es importante observar que"),
        (r"\bfijate que\b", "observa que"),
        (r"\bvamos a ver\b", "veremos"),
        (r"\bvamos a hacer\b", "realizaremos"),
        (r"\bvamos a crear\b", "crearemos"),
        (r"\bvamos a poner\b", "colocaremos"),
        (r"\bvamos a usar\b", "utilizaremos"),
        (r"\bun montón de\b", "muchos"),
        (r"\bacá\b", "aquí"),
        (r"\btenés\b", "tienes"),
        (r"\bpodés\b", "puedes"),
        (r"\bquerés\b", "quieres"),
        (r"\bdigamos,?\s*", ""),
        (r"\bbásicamente,?\s*", ""),
    ]
    for pat, rep in inversos:
        t = re.sub(pat, rep, t, flags=re.IGNORECASE)
    t = re.sub(r"\s+", " ", t).strip(" ,.-")
    if t:
        t = t[0].upper() + t[1:]
    return t


def generar_dataset_finetuning_jsonl(modelo: dict) -> str:
    """Exporta un archivo JSONL con pares (system, user=guión neutro, assistant=mi estilo real) para Fine-Tuning de LLMs."""
    bloques = modelo.get("bloques_rag", [])
    lineas = []
    sys_msg = "Sos un asistente que convierte guiones formales al estilo oral y natural de la autora."
    for b in bloques:
        real = b.get("texto", "").strip()
        if len(real.split()) < 12:
            continue
        neutro = _neutralizar_para_dataset(real)
        obj = {
            "messages": [
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": f"Convertí este guión a mi estilo de habla:\n{neutro}"},
                {"role": "assistant", "content": real}
            ]
        }
        lineas.append(json.dumps(obj, ensure_ascii=False))
    return "\n".join(lineas)


# ==============================================================================
# GESTIÓN DE GRUPOS DE HABLANTES (HOY: GRUPO CON NOMBRE / FUTURO: JERARQUÍA MULTINIVEL)
# ==============================================================================

def _asegurar_estructura_grupos(reg: dict) -> dict:
    """Garantiza que voces_registradas.json tenga soporte de grupos manteniendo compatibilidad."""
    reg.setdefault("perfiles", {})
    reg.setdefault("listado_nombres", [])
    grupos = reg.setdefault("grupos", {})
    if "General" not in grupos:
        grupos["General"] = {
            "nombre": "General",
            "categoria": "",
            "area": "",
            "subgrupo": "",
            "nombres": list(reg.get("listado_nombres", [])),
            "perfiles": dict(reg.get("perfiles", {})),
            "creado_en": time.strftime("%Y-%m-%d %H:%M:%S")
        }
    return reg


def listar_grupos() -> list[dict]:
    reg = _asegurar_estructura_grupos(cargar_registro_voces())
    salida = []
    for g_nom, g_info in reg.get("grupos", {}).items():
        perfiles_g = g_info.get("perfiles", {})
        nombres_g = list(g_info.get("nombres", []))
        for p_nom in perfiles_g.keys():
            if p_nom not in nombres_g and not p_nom.startswith("Mi voz (Hablante"):
                nombres_g.append(p_nom)
        salida.append({
            "nombre": g_nom,
            "categoria": g_info.get("categoria", ""),
            "area": g_info.get("area", ""),
            "subgrupo": g_info.get("subgrupo", ""),
            "cantidad_nombres": len(nombres_g),
            "cantidad_voces": len(perfiles_g),
            "nombres": sorted(nombres_g, key=lambda x: x.lower()),
            "perfiles": [
                {
                    "nombre": k,
                    "color": v.get("color", "#2563eb"),
                    "n_muestras": v.get("n_muestras", 1),
                    "origen": v.get("origen", "manual")
                }
                for k, v in perfiles_g.items()
            ]
        })
    return salida


def crear_o_actualizar_grupo(nombre_grupo: str, categoria: str = "", area: str = "", subgrupo: str = "") -> dict:
    nombre_grupo = (nombre_grupo or "").strip() or "General"
    reg = _asegurar_estructura_grupos(cargar_registro_voces())
    grupos = reg["grupos"]
    if nombre_grupo not in grupos:
        grupos[nombre_grupo] = {
            "nombre": nombre_grupo,
            "categoria": categoria.strip(),
            "area": area.strip(),
            "subgrupo": subgrupo.strip(),
            "nombres": [],
            "perfiles": {},
            "creado_en": time.strftime("%Y-%m-%d %H:%M:%S")
        }
    else:
        if categoria:
            grupos[nombre_grupo]["categoria"] = categoria.strip()
        if area:
            grupos[nombre_grupo]["area"] = area.strip()
        if subgrupo:
            grupos[nombre_grupo]["subgrupo"] = subgrupo.strip()
    guardar_registro_voces(reg)
    return grupos[nombre_grupo]


def registrar_voz_en_grupo(nombre_grupo: str, nombre_hablante: str, vector_emb: np.ndarray | None,
                           color: str = "#2563eb", origen: str = "manual") -> dict:
    nombre_grupo = (nombre_grupo or "").strip() or "General"
    nombre_hablante = (nombre_hablante or "").strip()
    if not nombre_hablante:
        raise ValueError("El nombre del hablante no puede estar vacío")

    reg = _asegurar_estructura_grupos(cargar_registro_voces())
    if nombre_grupo not in reg["grupos"]:
        reg["grupos"][nombre_grupo] = {
            "nombre": nombre_grupo, "categoria": "", "area": "", "subgrupo": "",
            "nombres": [], "perfiles": {}, "creado_en": time.strftime("%Y-%m-%d %H:%M:%S")
        }
    g = reg["grupos"][nombre_grupo]
    if nombre_hablante not in g["nombres"]:
        g["nombres"].append(nombre_hablante)

    if vector_emb is not None:
        perfiles = g.setdefault("perfiles", {})
        emb_lista = [float(x) for x in vector_emb.flatten()]
        if nombre_hablante in perfiles and "embedding" in perfiles[nombre_hablante]:
            emb_ant = np.array(perfiles[nombre_hablante]["embedding"], dtype=np.float32)
            m_ant = perfiles[nombre_hablante].get("n_muestras", 1)
            nuevo = (emb_ant * m_ant + vector_emb) / float(m_ant + 1)
            perfiles[nombre_hablante] = {
                "nombre": nombre_hablante,
                "embedding": [float(x) for x in nuevo.flatten()],
                "color": color or perfiles[nombre_hablante].get("color", "#2563eb"),
                "n_muestras": m_ant + 1,
                "origen": origen,
                "actualizado_en": time.strftime("%Y-%m-%d %H:%M:%S")
            }
        else:
            perfiles[nombre_hablante] = {
                "nombre": nombre_hablante,
                "embedding": emb_lista,
                "color": color or "#2563eb",
                "n_muestras": 1,
                "origen": origen,
                "creado_en": time.strftime("%Y-%m-%d %H:%M:%S")
            }
        # También espejar en el diccionario global para compatibilidad directa
        reg["perfiles"][nombre_hablante] = perfiles[nombre_hablante]

    if nombre_hablante not in reg.get("listado_nombres", []):
        reg.setdefault("listado_nombres", []).append(nombre_hablante)

    guardar_registro_voces(reg)
    return g


# ==============================================================================
# PERSISTENCIA DE SESIONES DE TRANSCRIPCIÓN EN DISCO
# ==============================================================================

def guardar_sesion_disco(sesion_data: dict) -> None:
    """Guarda la sesión de transcripción en transcripciones/ para no perderla al reiniciar."""
    try:
        sid = sesion_data["sesion_id"]
        copia = {k: v for k, v in sesion_data.items() if k != "X"}
        ruta_json = os.path.join(SESIONES_DIR, f"{sid}_sesion.json")
        with open(ruta_json, "w", encoding="utf-8") as f:
            json.dump(copia, f, ensure_ascii=False, indent=2)
        X = sesion_data.get("X")
        if X is not None and isinstance(X, np.ndarray) and len(X) > 0:
            ruta_npy = os.path.join(SESIONES_DIR, f"{sid}_X.npy")
            np.save(ruta_npy, X)
    except Exception as e:
        print(f"Aviso: no se pudo guardar sesión en disco: {e}")


def cargar_sesiones_disco() -> None:
    """Carga en memoria las sesiones guardadas previamente en transcripciones/."""
    if not os.path.isdir(SESIONES_DIR):
        return
    for fn in os.listdir(SESIONES_DIR):
        if not fn.endswith("_sesion.json"):
            continue
        ruta_json = os.path.join(SESIONES_DIR, fn)
        try:
            with open(ruta_json, "r", encoding="utf-8") as f:
                data = json.load(f)
            sid = data.get("sesion_id")
            if not sid:
                sid = fn.replace("_sesion.json", "")
                data["sesion_id"] = sid
            nombre_mp3 = data.get("nombre_mp3", "")
            ruta_mp3 = os.path.join(SALIDA, nombre_mp3) if nombre_mp3 else ""
            data.setdefault("ruta_mp3", ruta_mp3)
            data.setdefault("nombre_base", os.path.splitext(data.get("nombre_original", "audio"))[0])
            data.setdefault("cal", "192k")
            data.setdefault("mono", True)
            data.setdefault("grupo", "General")
            data.setdefault("offset", 0.0)
            if "sincronizado" not in data:
                es_whisper = not str(data.get("modelo", "")).lower().startswith("importado")
                data["sincronizado"] = bool(es_whisper and data.get("nombre_mp3"))
                if data["sincronizado"]:
                    data["origen_media"] = "whisper"

            ruta_npy = os.path.join(SESIONES_DIR, f"{sid}_X.npy")
            if os.path.isfile(ruta_npy):
                data["X"] = np.load(ruta_npy)
            else:
                data["X"] = None

            with JOBS_LOCK:
                SESIONES[sid] = data
        except Exception as e:
            print(f"Error cargando sesión {fn}: {e}")


# ==============================================================================
# LISTADO DE NOMBRES DE HABLANTES (EXCEL .XLSX, .CSV, .TXT) POR GRUPO
# ==============================================================================

def obtener_listado_nombres(grupo: str = "General") -> list[str]:
    reg = _asegurar_estructura_grupos(cargar_registro_voces())
    g = reg["grupos"].get(grupo) or reg["grupos"].get("General", {})
    lista = list(g.get("nombres", []))
    for nom in g.get("perfiles", {}).keys():
        if nom not in lista and not nom.startswith("Mi voz (Hablante"):
            lista.append(nom)
    return sorted(lista, key=lambda x: x.lower())


def agregar_nombres_al_listado(nuevos: list[str], grupo: str = "General", reemplazar: bool = False) -> list[str]:
    grupo = (grupo or "").strip() or "General"
    reg = _asegurar_estructura_grupos(cargar_registro_voces())
    if grupo not in reg["grupos"]:
        reg["grupos"][grupo] = {
            "nombre": grupo, "categoria": "", "area": "", "subgrupo": "",
            "nombres": [], "perfiles": {}, "creado_en": time.strftime("%Y-%m-%d %H:%M:%S")
        }
    g = reg["grupos"][grupo]
    actuales = [] if reemplazar else list(g.get("nombres", []))
    vistos = {x.lower(): x for x in actuales}
    for n in nuevos:
        limpio = re.sub(r"\s+", " ", (n or "").strip())
        if len(limpio) >= 2 and limpio.lower() not in vistos:
            vistos[limpio.lower()] = limpio
            actuales.append(limpio)
    g["nombres"] = actuales
    reg["listado_nombres"] = actuales
    guardar_registro_voces(reg)
    return obtener_listado_nombres(grupo)


def extraer_nombres_desde_archivo(datos_bytes: bytes, nombre_archivo: str) -> list[str]:
    """Extrae nombres de hablantes desde un archivo Excel (.xlsx), .csv o .txt."""
    ext = os.path.splitext(nombre_archivo or "")[1].lower()
    ignorar = {
        "nombre", "nombres", "apellido", "apellidos", "nombre y apellido",
        "apellido y nombre", "hablante", "hablantes",
        "participante", "participantes", "persona", "email", "correo",
        "dni", "legajo", "id", "nro", "numero", "equipo", "grupo", "rol"
    }
    candidatos = []

    if ext == ".xlsx":
        with zipfile.ZipFile(io.BytesIO(datos_bytes)) as zf:
            shared_strings = []
            if "xl/sharedStrings.xml" in zf.namelist():
                root_ss = ET.fromstring(zf.read("xl/sharedStrings.xml"))
                for si in root_ss.iter():
                    if si.tag.endswith("}si") or si.tag == "si":
                        partes = [t.text or "" for t in si.iter() if (t.tag.endswith("}t") or t.tag == "t")]
                        shared_strings.append("".join(partes).strip())

            hojas = [n for n in zf.namelist() if n.startswith("xl/worksheets/sheet") and n.endswith(".xml")]
            for hoja in sorted(hojas):
                root_sh = ET.fromstring(zf.read(hoja))
                for row in root_sh.iter():
                    if not (row.tag.endswith("}row") or row.tag == "row"):
                        continue
                    valores_fila = []
                    for c in row:
                        if not (c.tag.endswith("}c") or c.tag == "c"):
                            continue
                        tipo = c.attrib.get("t", "")
                        val_txt = ""
                        if tipo == "s":
                            for v in c:
                                if (v.tag.endswith("}v") or v.tag == "v") and v.text and v.text.isdigit():
                                    idx = int(v.text)
                                    if 0 <= idx < len(shared_strings):
                                        val_txt = shared_strings[idx]
                        elif tipo == "inlineStr":
                            partes = [t.text or "" for t in c.iter() if (t.tag.endswith("}t") or t.tag == "t")]
                            val_txt = "".join(partes).strip()
                        else:
                            for v in c:
                                if (v.tag.endswith("}v") or v.tag == "v") and v.text:
                                    val_txt = v.text.strip()
                        if val_txt:
                            valores_fila.append(val_txt)
                    textos_validos = [
                        v.strip() for v in valores_fila
                        if v.strip() and not re.fullmatch(r"[\d.,:/\-]+", v.strip())
                        and "@" not in v and v.strip().lower() not in ignorar
                    ]
                    if len(textos_validos) == 1:
                        candidatos.append(textos_validos[0])
                    elif len(textos_validos) >= 2:
                        a, b = textos_validos[0], textos_validos[1]
                        if len(a.split()) <= 2 and len(b.split()) <= 2:
                            candidatos.append(f"{a} {b}")
                        else:
                            candidatos.extend(textos_validos[:2])
    else:
        try:
            texto = datos_bytes.decode("utf-8-sig")
        except UnicodeDecodeError:
            texto = datos_bytes.decode("latin-1", errors="replace")

        for linea in texto.splitlines():
            linea = linea.strip()
            if not linea:
                continue
            sep = ";" if ";" in linea else ("," if "," in linea else "\t")
            if sep in linea and ext == ".csv":
                cols = [c.strip().strip('"').strip("'") for c in linea.split(sep)]
                cols_txt = [
                    c for c in cols
                    if c and not re.fullmatch(r"[\d.,:/\-]+", c) and "@" not in c and c.lower() not in ignorar
                ]
                if len(cols_txt) == 1:
                    candidatos.append(cols_txt[0])
                elif len(cols_txt) >= 2:
                    if len(cols_txt[0].split()) <= 2 and len(cols_txt[1].split()) <= 2:
                        candidatos.append(f"{cols_txt[0]} {cols_txt[1]}")
                    else:
                        candidatos.append(cols_txt[0])
            else:
                limpio = linea.strip().strip("-•*0123456789.) ").strip()
                if limpio and limpio.lower() not in ignorar and not re.fullmatch(r"[\d.,:/\-]+", limpio):
                    candidatos.append(limpio)

    resultado = []
    vistos = set()
    for c in candidatos:
        c_norm = re.sub(r"\s+", " ", c).strip()
        if len(c_norm) >= 2 and c_norm.lower() not in ignorar and c_norm.lower() not in vistos:
            vistos.add(c_norm.lower())
            resultado.append(c_norm)
    return resultado


# ==============================================================================
# AUTO-DETECTOR Y PARSER DE TRANSCRIPCIONES EXTERNAS (TACTIQ, YOUTUBE, WHISPER, SRT)
# ==============================================================================

def _ts_a_seg(ts: str) -> float:
    ts = ts.strip().replace(",", ".")
    partes = ts.split(":")
    try:
        if len(partes) == 2:
            return int(partes[0]) * 60 + float(partes[1])
        if len(partes) == 3:
            return int(partes[0]) * 3600 + int(partes[1]) * 60 + float(partes[2])
        return float(ts)
    except Exception:
        return 0.0


def _extraer_texto_documento(datos_bytes: bytes, nombre_archivo: str) -> str:
    ext = os.path.splitext(nombre_archivo or "")[1].lower()
    if ext == ".docx":
        with zipfile.ZipFile(io.BytesIO(datos_bytes)) as zf:
            xml_content = zf.read("word/document.xml")
        root = ET.fromstring(xml_content)
        parrafos = []
        for p in root.iter():
            if p.tag.endswith("}p") or p.tag == "p":
                textos = [t.text for t in p.iter() if (t.tag.endswith("}t") or t.tag == "t") and t.text]
                if textos:
                    parrafos.append("".join(textos))
        return "\n".join(parrafos)
    if ext == ".pdf":
        for mod_name in ("pypdf", "PyPDF2"):
            try:
                mod = __import__(mod_name)
                reader = mod.PdfReader(io.BytesIO(datos_bytes))
                return "\n".join((page.extract_text() or "") for page in reader.pages)
            except Exception:
                pass
    try:
        return datos_bytes.decode("utf-8-sig")
    except UnicodeDecodeError:
        return datos_bytes.decode("latin-1", errors="replace")


def parsear_transcripcion_externa(datos_bytes: bytes, nombre_archivo: str) -> dict:
    """Detecta automáticamente si una transcripción es de Tactiq, YouTube, Whisper o SRT/VTT
    y extrae sus segmentos con tiempos y hablantes."""
    texto_raw = _extraer_texto_documento(datos_bytes, nombre_archivo)
    lineas = [l.rstrip() for l in texto_raw.splitlines()]
    formato_detectado = "Texto general"
    items = []

    re_srt = re.compile(r"(\d{1,2}:\d{2}:\d{2}[,.]\d{1,3}|\d{1,2}:\d{2}[,.]\d{1,3})\s*-->\s*(\d{1,2}:\d{2}:\d{2}[,.]\d{1,3}|\d{1,2}:\d{2}[,.]\d{1,3})")
    re_sbv = re.compile(r"^(\d{1,2}:\d{2}:\d{2}\.\d{1,3}),(\d{1,2}:\d{2}:\d{2}\.\d{1,3})$")
    re_whisper = re.compile(r"^\[(\d{1,2}:\d{2}(?::\d{2})?(?:\.\d+)?)\s*(?:->|-|–|a)\s*(\d{1,2}:\d{2}(?::\d{2})?(?:\.\d+)?)\]\s*(.*)$")
    re_tactiq_linea = re.compile(r"^\[?(\d{1,2}:\d{2}(?::\d{2})?)\]?\s+(?:-\s+)?([^:]{1,45}):\s*(.+)$")
    re_tactiq_header1 = re.compile(r"^(\d{1,2}:\d{2}(?::\d{2})?)\s+([A-Za-zÁÉÍÓÚÜÑáéíóúüñ0-9_().\-\s]{2,45})$")
    re_tactiq_header2 = re.compile(r"^([A-Za-zÁÉÍÓÚÜÑáéíóúüñ0-9_().\-\s]{2,45})\s+(\d{1,2}:\d{2}(?::\d{2})?)$")
    re_yt_ts_solo = re.compile(r"^(\d{1,2}:\d{2}(?::\d{2})?)$")
    re_yt_ts_linea = re.compile(r"^(\d{1,2}:\d{2}(?::\d{2})?)\s+(.+)$")

    n_srt = sum(1 for l in lineas if re_srt.search(l))
    n_sbv = sum(1 for l in lineas if re_sbv.match(l.strip()))
    n_whisper = sum(1 for l in lineas if re_whisper.match(l.strip()))
    n_tactiq_l = sum(1 for l in lineas if re_tactiq_linea.match(l.strip()))
    tiene_marca_tactiq = "tactiq" in texto_raw.lower() or "google meet" in texto_raw.lower()

    if n_srt >= 2 or n_sbv >= 2:
        formato_detectado = "Subtítulos (SRT / VTT / YouTube)"
        i = 0
        while i < len(lineas):
            l = lineas[i].strip()
            m1 = re_srt.search(l)
            m2 = re_sbv.match(l) if not m1 else None
            if m1 or m2:
                ini = _ts_a_seg(m1.group(1) if m1 else m2.group(1))
                fin = _ts_a_seg(m1.group(2) if m1 else m2.group(2))
                i += 1
                txt_partes = []
                while i < len(lineas) and lineas[i].strip() and not re_srt.search(lineas[i]) and not re_sbv.match(lineas[i].strip()):
                    if not lineas[i].strip().isdigit():
                        txt_partes.append(lineas[i].strip())
                    i += 1
                txt_unido = " ".join(txt_partes).strip()
                if txt_unido:
                    spk_nom = "Hablante 1"
                    m_spk = re.match(r"^\[([^\]]{1,40})\]\s*(.+)$", txt_unido) or re.match(r"^([^:]{2,35}):\s+(.+)$", txt_unido)
                    if m_spk:
                        spk_nom = m_spk.group(1).strip()
                        txt_unido = m_spk.group(2).strip()
                    items.append({"inicio": ini, "fin": fin, "hablante_nombre": spk_nom, "texto": txt_unido})
            else:
                i += 1

    elif n_whisper >= 2:
        formato_detectado = "Whisper"
        for l in lineas:
            m = re_whisper.match(l.strip())
            if m:
                ini = _ts_a_seg(m.group(1))
                fin = _ts_a_seg(m.group(2))
                resto = m.group(3).strip()
                spk_nom = "Hablante 1"
                m_spk = re.match(r"^([^:]{1,40}):\s*(.+)$", resto)
                if m_spk:
                    spk_nom = m_spk.group(1).strip()
                    resto = m_spk.group(2).strip()
                if resto:
                    items.append({"inicio": ini, "fin": fin, "hablante_nombre": spk_nom, "texto": resto})

    elif n_tactiq_l >= 2:
        formato_detectado = "Tactiq"
        for l in lineas:
            m = re_tactiq_linea.match(l.strip())
            if m:
                ini = _ts_a_seg(m.group(1))
                spk_nom = m.group(2).strip()
                txt = m.group(3).strip()
                if txt:
                    items.append({"inicio": ini, "fin": None, "hablante_nombre": spk_nom, "texto": txt})
            elif items and l.strip() and not l.startswith("#") and "tactiq.io" not in l.lower():
                items[-1]["texto"] += " " + l.strip()

    else:
        i = 0
        usos_hablante = 0
        while i < len(lineas):
            l = lineas[i].strip()
            if not l or "tactiq.io" in l.lower() or l.lower().startswith("transcript"):
                i += 1
                continue
            m_ys = re_yt_ts_solo.match(l)
            m_th1 = re_tactiq_header1.match(l) if not m_ys else None
            m_th2 = re_tactiq_header2.match(l) if not (m_ys or m_th1) else None
            m_yl = re_yt_ts_linea.match(l) if not (m_ys or m_th1 or m_th2) else None

            if m_ys:
                ini = _ts_a_seg(m_ys.group(1))
                i += 1
                txt_partes = []
                while i < len(lineas) and lineas[i].strip() and not re_yt_ts_solo.match(lineas[i].strip()):
                    txt_partes.append(lineas[i].strip())
                    i += 1
                txt = " ".join(txt_partes).strip()
                if txt:
                    spk_nom = "Hablante 1"
                    m_spk = re.match(r"^([^:]{2,35}):\s+(.+)$", txt)
                    if m_spk:
                        spk_nom = m_spk.group(1).strip()
                        txt = m_spk.group(2).strip()
                        usos_hablante += 1
                    items.append({"inicio": ini, "fin": None, "hablante_nombre": spk_nom, "texto": txt})
            elif m_th1 or m_th2:
                ini = _ts_a_seg(m_th1.group(1) if m_th1 else m_th2.group(2))
                spk_nom = (m_th1.group(2) if m_th1 else m_th2.group(1)).strip()
                usos_hablante += 1
                i += 1
                txt_partes = []
                while i < len(lineas) and lineas[i].strip() and not re_tactiq_header1.match(lineas[i].strip()) and not re_tactiq_header2.match(lineas[i].strip()):
                    txt_partes.append(lineas[i].strip())
                    i += 1
                txt = " ".join(txt_partes).strip()
                if txt:
                    items.append({"inicio": ini, "fin": None, "hablante_nombre": spk_nom, "texto": txt})
            elif m_yl:
                ini = _ts_a_seg(m_yl.group(1))
                txt = m_yl.group(2).strip()
                items.append({"inicio": ini, "fin": None, "hablante_nombre": "Hablante 1", "texto": txt})
                i += 1
            else:
                i += 1

        if items:
            if tiene_marca_tactiq or usos_hablante >= 2:
                formato_detectado = "Tactiq"
            else:
                formato_detectado = "YouTube"
        else:
            formato_detectado = "Texto sin tiempos (tiempos estimados)"
            cursor = 0.0
            for l in lineas:
                l_str = l.strip()
                if not l_str:
                    continue
                spk_nom = "Hablante 1"
                m_spk = re.match(r"^([^:]{2,35}):\s+(.+)$", l_str)
                if m_spk:
                    spk_nom = m_spk.group(1).strip()
                    l_str = m_spk.group(2).strip()
                dur_est = max(2.0, round(len(l_str) / 14.0, 1))
                items.append({"inicio": round(cursor, 2), "fin": round(cursor + dur_est, 2), "hablante_nombre": spk_nom, "texto": l_str})
                cursor += dur_est + 0.3

    for idx, it in enumerate(items):
        if it["fin"] is None or it["fin"] <= it["inicio"]:
            dur_est = max(1.5, min(15.0, round(len(it["texto"]) / 13.5, 2)))
            if idx + 1 < len(items) and items[idx + 1]["inicio"] > it["inicio"]:
                it["fin"] = round(min(items[idx + 1]["inicio"], it["inicio"] + dur_est + 2.0), 2)
            else:
                it["fin"] = round(it["inicio"] + dur_est, 2)

    mapa_nom_a_id = {}
    nombres_previos = {}
    segmentos = []
    for idx, it in enumerate(items):
        nom = it["hablante_nombre"] or "Hablante 1"
        if nom not in mapa_nom_a_id:
            spk_id = f"SPK_{len(mapa_nom_a_id) + 1}"
            mapa_nom_a_id[nom] = spk_id
            nombres_previos[spk_id] = nom
        spk_id = mapa_nom_a_id[nom]
        segmentos.append({
            "id": idx,
            "inicio": round(float(it["inicio"]), 2),
            "fin": round(float(it["fin"]), 2),
            "texto": it["texto"],
            "hablante": spk_id,
            "confianza": 0.95,
            "silenciado": False
        })

    duracion_total = max((s["fin"] for s in segmentos), default=0.0)
    hablantes = _resumen_hablantes(segmentos, nombres_previos=nombres_previos)
    return {
        "formato_detectado": formato_detectado,
        "segmentos": segmentos,
        "hablantes": hablantes,
        "duracion": round(duracion_total, 2)
    }


# ==============================================================================
# AUTO-MATCHEO INICIAL DE HABLANTES: PRESENTACIONES ("MI NOMBRE ES / SOY...") + CRUCE CON TACTIQ SINCRONIZADO
# ==============================================================================

def _matchear_nombre_con_listado(candidato: str, listado_nombres: list[str]) -> str:
    """Si hay un listado de nombres del grupo (ej. Excel), empareja 'Sofi' o 'Sofía' con 'Sofía Gómez'."""
    cand_norm = normalizar_texto(candidato)
    if not cand_norm:
        return candidato.strip().title()
    if not listado_nombres:
        return candidato.strip().title()

    mejor_nom = None
    mejor_score = 0.0
    partes_cand = cand_norm.split()
    for nom_lista in listado_nombres:
        nl_norm = normalizar_texto(nom_lista)
        if not nl_norm:
            continue
        if cand_norm == nl_norm:
            return nom_lista
        partes_nl = nl_norm.split()
        # Si el primer nombre coincide exacto
        if partes_cand and partes_nl and partes_cand[0] == partes_nl[0]:
            score = 0.92
        elif cand_norm in nl_norm or nl_norm in cand_norm:
            score = 0.88
        else:
            score = difflib.SequenceMatcher(None, cand_norm, nl_norm).ratio()
        if score > mejor_score:
            mejor_score = score
            mejor_nom = nom_lista

    if mejor_nom and mejor_score >= 0.72:
        return mejor_nom
    return candidato.strip().title()


def detectar_presentaciones_en_segmentos(segmentos: list[dict], listado_nombres: list[str]) -> list[dict]:
    """Analiza qué dice cada hablante buscando frases de presentación ('mi nombre es...', 'soy...', 'me llamo...')
    y cruza con el listado de nombres del grupo."""
    patrones = [
        re.compile(r"\b(?:mi nombre es|me llamo|yo soy|hola soy|buenas soy|buen dia soy|buenas tardes soy|buenas noches soy|aca habla)\s+([a-záéíóúüñ]{3,20}(?:\s+[a-záéíóúüñ]{3,20})?)", re.IGNORECASE),
    ]
    no_nombres = {
        "nuevo", "nueva", "integrante", "miembro", "participante", "coordinador",
        "de", "del", "la", "el", "un", "una", "yo", "muy", "medio", "bastante", "parte",
        "aca", "ahi", "bien", "todo", "toda", "como", "porque", "cuando", "quien"
    }

    hallazgos = []
    for s in segmentos:
        txt = s.get("texto", "")
        txt_norm = normalizar_texto(txt)
        encontrado = None
        metodo = ""

        for pat in patrones:
            m = pat.search(txt)
            if m:
                bruto = m.group(1).strip()
                palabras_b = [p for p in bruto.split() if p.lower() not in no_nombres]
                if palabras_b:
                    candidato = " ".join(palabras_b[:2])
                    encontrado = _matchear_nombre_con_listado(candidato, listado_nombres)
                    metodo = f'Dijo en el audio: "{txt[:70]}"'
                    break

        # Si no usó la frase exacta pero hay listado de nombres y el segmento es corto de presentación
        if not encontrado and listado_nombres and len(txt_norm.split()) <= 18:
            if any(p in txt_norm for p in ("hola", "buenas", "presento", "presentarme", "nombre", "presente", "aca estoy")):
                for nom_l in listado_nombres:
                    p_nom = normalizar_texto(nom_l).split()
                    if p_nom and len(p_nom[0]) >= 3 and re.search(rf"\b{re.escape(p_nom[0])}\b", txt_norm):
                        encontrado = nom_l
                        metodo = f'Mencionó su nombre del listado en: "{txt[:70]}"'
                        break

        if encontrado:
            hallazgos.append({
                "seg_id": s.get("id", 0),
                "inicio": s["inicio"],
                "fin": s["fin"],
                "spk_id": s.get("hablante", "SPK_1"),
                "nombre_detectado": encontrado,
                "motivo": metodo,
                "texto": txt
            })
    return hallazgos


def sincronizar_y_matchear_con_tactiq(segmentos_audio: list[dict], segmentos_tactiq: list[dict],
                                      hablantes_tactiq: list[dict], listado_nombres: list[str]) -> dict:
    """1. Calcula automáticamente el desfase de tiempo (offset) entre la transcripción de Tactiq y el audio
       comparando frases similares.
       2. Asigna a cada segmento/hablante del audio el nombre correspondiente de Tactiq (cruzado con el listado)."""
    mapa_id_a_nom_tactiq = {h["id"]: h["nombre"] for h in hablantes_tactiq}

    # Paso 1: Estimar desfase (offset = t_audio - t_tactiq) buscando frases en común
    desfases = []
    for st in segmentos_tactiq:
        txt_t = normalizar_texto(st.get("texto", ""))
        if len(txt_t.split()) < 4:
            continue
        mejor_r = 0.0
        mejor_sa = None
        for sa in segmentos_audio:
            txt_a = normalizar_texto(sa.get("texto", ""))
            if len(txt_a.split()) < 4:
                continue
            r = difflib.SequenceMatcher(None, txt_t[:80], txt_a[:80]).ratio()
            if r > mejor_r:
                mejor_r = r
                mejor_sa = sa
        if mejor_sa and mejor_r >= 0.65:
            desfases.append(float(mejor_sa["inicio"]) - float(st["inicio"]))

    offset_detectado = float(np.median(desfases)) if desfases else 0.0

    # Paso 2: Con el offset aplicado, asignar hablante de Tactiq a cada segmento de audio
    votos_por_spk = {}  # spk_audio -> {nombre_oficial: segundos_solapados}
    asignaciones_seg = 0

    for sa in segmentos_audio:
        a_ini = float(sa["inicio"])
        a_fin = float(sa["fin"])
        mejor_nom = None
        mejor_peso = 0.0

        for st in segmentos_tactiq:
            t_ini = float(st["inicio"]) + offset_detectado
            t_fin = float(st["fin"]) + offset_detectado
            solape = min(a_fin, t_fin + 1.5) - max(a_ini, t_ini - 1.5)
            if solape > 0:
                sim_txt = difflib.SequenceMatcher(
                    None, normalizar_texto(sa.get("texto", ""))[:60], normalizar_texto(st.get("texto", ""))[:60]
                ).ratio()
                peso = solape * (1.0 + sim_txt * 2.0)
                if peso > mejor_peso:
                    mejor_peso = peso
                    nom_bruto = mapa_id_a_nom_tactiq.get(st.get("hablante"), "Hablante 1")
                    if not nom_bruto.startswith("Hablante "):
                        mejor_nom = _matchear_nombre_con_listado(nom_bruto, listado_nombres)

        if mejor_nom:
            spk_a = sa.get("hablante", "SPK_1")
            votos_por_spk.setdefault(spk_a, {})
            dur_s = max(0.5, a_fin - a_ini)
            votos_por_spk[spk_a][mejor_nom] = votos_por_spk[spk_a].get(mejor_nom, 0.0) + dur_s
            asignaciones_seg += 1

    mapeo_spk = {}
    for spk_a, conteo_noms in votos_por_spk.items():
        if conteo_noms:
            ganador = max(conteo_noms.items(), key=lambda x: x[1])[0]
            mapeo_spk[spk_a] = ganador

    return {
        "offset_segundos": round(offset_detectado, 2),
        "coincidencias_frases": len(desfases),
        "segmentos_emparejados": asignaciones_seg,
        "mapeo_hablantes": mapeo_spk
    }


HTML = r"""<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Estudio de Audio</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@500;600&display=swap" rel="stylesheet">
<style>
/* ==========================================================================
   SISTEMA VISUAL · claro, sobrio, con barra lateral
   ========================================================================== */
:root {
  --bg: #f5f6f8;
  --card: #ffffff;
  --surface-2: #f8f9fb;
  --text: #111827;
  --text-2: #374151;
  --muted: #6b7280;
  --border: #e5e7eb;
  --border-strong: #d1d5db;
  --primary: #2563eb;
  --primary-hover: #1d4ed8;
  --primary-light: #eff4ff;
  --primary-border: #c7d7fe;
  --secondary: #0f766e;
  --secondary-hover: #115e59;
  --danger: #dc2626;
  --danger-bg: #fef2f2;
  --ok: #16a34a;
  --ok-bg: #f0fdf4;
  --warn: #d97706;
  --warn-bg: #fffbeb;
  --radius: 12px;
  --radius-sm: 8px;
  --shadow-sm: 0 1px 2px rgba(17,24,39,0.04);
  --shadow-md: 0 4px 16px rgba(17,24,39,0.08);
  --sidebar-w: 248px;
  --font: "Inter", "Segoe UI", system-ui, -apple-system, Roboto, sans-serif;
  --mono: "JetBrains Mono", ui-monospace, "Cascadia Mono", Consolas, monospace;
}
* { box-sizing: border-box; }
html { scroll-behavior: smooth; }
body {
  margin: 0;
  font-family: var(--font);
  font-size: 14.5px;
  background: var(--bg);
  color: var(--text);
  line-height: 1.5;
  -webkit-font-smoothing: antialiased;
}
code {
  font-family: var(--mono);
  font-size: 0.86em;
  background: #f1f3f6;
  padding: 1px 5px;
  border-radius: 4px;
  color: var(--text-2);
}

/* ---------- Estructura general ---------- */
.app {
  display: grid;
  grid-template-columns: var(--sidebar-w) 1fr;
  min-height: 100vh;
}
.sidebar {
  position: sticky;
  top: 0;
  height: 100vh;
  background: #fbfbfc;
  border-right: 1px solid var(--border);
  display: flex;
  flex-direction: column;
  padding: 18px 14px;
  gap: 18px;
  overflow-y: auto;
}
.brand {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 2px 6px 4px;
}
.brand-logo {
  width: 34px; height: 34px;
  border-radius: 9px;
  background: var(--primary);
  color: #fff;
  display: grid; place-items: center;
  flex-shrink: 0;
}
.brand-logo svg { width: 18px; height: 18px; }
.brand-name { font-weight: 700; font-size: 0.98rem; line-height: 1.2; }
.brand-sub { font-size: 0.74rem; color: var(--muted); }

.nav-group-label {
  font-size: 0.7rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: #9ca3af;
  padding: 0 10px;
  margin: 4px 0 6px;
}
.pasos-stepper { display: flex; flex-direction: column; gap: 2px; }
.step-btn {
  all: unset;
  box-sizing: border-box;
  display: grid;
  grid-template-columns: 26px 1fr;
  align-items: center;
  gap: 4px 10px;
  width: 100%;
  padding: 9px 10px;
  border-radius: 8px;
  cursor: pointer;
  color: var(--text-2);
  transition: background 0.12s, color 0.12s;
  position: relative;
}
.step-btn:hover { background: #f0f2f5; }
.step-btn:focus-visible { outline: 2px solid var(--primary); outline-offset: 1px; }
.step-btn .step-num {
  width: 24px; height: 24px;
  border-radius: 7px;
  display: grid; place-items: center;
  font-size: 0.74rem; font-weight: 700;
  background: #eceff3;
  color: var(--muted);
  grid-row: span 2;
}
.step-btn .step-num svg { width: 14px; height: 14px; }
.step-btn .step-title { font-size: 0.88rem; font-weight: 600; line-height: 1.25; }
.step-btn .step-sub { font-size: 0.74rem; color: var(--muted); line-height: 1.3; }
.step-btn.activo { background: var(--primary-light); color: var(--primary-hover); }
.step-btn.activo .step-num { background: var(--primary); color: #fff; }
.step-btn.activo::before {
  content: ""; position: absolute; left: -14px; top: 8px; bottom: 8px;
  width: 3px; border-radius: 0 3px 3px 0; background: var(--primary);
}

.sidebar-foot { margin-top: auto; display: flex; flex-direction: column; gap: 8px; }
.side-link {
  all: unset; box-sizing: border-box;
  display: flex; align-items: center; gap: 8px;
  padding: 8px 10px; border-radius: 8px;
  font-size: 0.84rem; font-weight: 500; color: var(--text-2);
  cursor: pointer;
}
.side-link:hover { background: #f0f2f5; }
.side-link svg { width: 16px; height: 16px; color: var(--muted); }
#git-bar {
  background: #fff;
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 10px;
  font-size: 0.78rem;
  display: flex;
  flex-direction: column;
  gap: 8px;
}
#git-bar .git-head { display: flex; align-items: center; gap: 6px; color: var(--muted); font-weight: 600; font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.05em; }
#git-bar .git-head svg { width: 13px; height: 13px; }
#git-estado-txt { font-weight: 600; font-size: 0.8rem; color: var(--text-2); }
#git-bar .git-acciones { display: flex; gap: 6px; }
#git-bar .git-acciones button { flex: 1; justify-content: center; }

.main {
  min-width: 0;
  padding: 28px 32px 80px;
}
.main-inner { max-width: 1160px; margin: 0 auto; }

/* ---------- Encabezado de cada sección ---------- */
.sec-head {
  display: flex;
  justify-content: space-between;
  align-items: flex-end;
  gap: 16px;
  flex-wrap: wrap;
  margin-bottom: 20px;
}
.sec-head .eyebrow {
  font-size: 0.74rem; font-weight: 600; color: var(--primary);
  text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 4px;
}
.sec-head h1 { margin: 0; font-size: 1.5rem; font-weight: 700; letter-spacing: -0.015em; }
.sec-head p { margin: 4px 0 0; color: var(--muted); font-size: 0.92rem; max-width: 720px; }

.step-section { display: none; animation: aparecer 0.18s ease; }
.step-section.activo { display: block; }
@keyframes aparecer { from { opacity: 0; transform: translateY(4px); } to { opacity: 1; transform: none; } }

/* ---------- "Usalo para" ---------- */
.uso {
  display: flex; gap: 14px; align-items: baseline; flex-wrap: wrap;
  margin: -6px 0 18px; padding: 10px 14px;
  background: #fff; border: 1px solid var(--border); border-radius: 10px;
}
.uso-tit { font-size: 0.72rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em; color: var(--primary); white-space: nowrap; }
.uso ul { list-style: none; margin: 0; padding: 0; display: flex; flex-wrap: wrap; gap: 4px 18px; }
.uso li { font-size: 0.84rem; color: var(--text-2); position: relative; padding-left: 16px; }
.uso li::before {
  content: ""; position: absolute; left: 2px; top: 0.45em; width: 7px; height: 4px;
  border-left: 2px solid var(--ok); border-bottom: 2px solid var(--ok); transform: rotate(-45deg);
}

/* ---------- Ejemplo en la ayuda ---------- */
.ejemplo { margin-top: 18px; border-top: 1px solid var(--border); padding-top: 16px; }
.ejemplo h3 { margin: 0 0 4px; font-size: 1rem; }
.ejemplo > p { margin: 0 0 12px; color: var(--muted); font-size: 0.86rem; }
.ejemplo ol { margin: 0; padding: 0; list-style: none; counter-reset: ej; display: flex; flex-direction: column; gap: 10px; }
.ejemplo li { counter-increment: ej; display: grid; grid-template-columns: 26px 1fr; gap: 10px; font-size: 0.86rem; color: var(--text-2); }
.ejemplo li::before {
  content: counter(ej); width: 24px; height: 24px; border-radius: 50%; background: var(--primary-light);
  color: var(--primary); font-weight: 700; font-size: 0.78rem; display: grid; place-items: center;
}
.ejemplo li strong { color: var(--text); }
.ejemplo li em { color: var(--muted); font-style: normal; }

/* ---------- Tarjetas ---------- */
.card {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 20px 22px;
  margin-bottom: 16px;
  box-shadow: var(--shadow-sm);
}
.card h2 { font-size: 1.05rem; font-weight: 650; margin: 0 0 4px; color: var(--text); }
.card h3 { font-size: 0.98rem; font-weight: 650; margin: 0 0 4px; color: var(--text); }
.card-head {
  display: flex; justify-content: space-between; align-items: flex-start;
  gap: 12px; flex-wrap: wrap; margin-bottom: 14px;
}
.card-head p, .sub-desc { font-size: 0.86rem; color: var(--muted); margin: 2px 0 0; }
.sub-desc { margin-bottom: 14px; }
.card-tag {
  display: inline-block; font-size: 0.7rem; font-weight: 600;
  color: var(--muted); background: #f1f3f6; border-radius: 5px;
  padding: 1px 7px; margin-bottom: 6px; text-transform: uppercase; letter-spacing: 0.05em;
}
.card.destacada { border-color: var(--primary-border); background: linear-gradient(180deg, #f7f9ff 0%, #fff 100%); }

.panel {
  background: var(--surface-2);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 14px;
}
.panel-titulo { font-size: 0.86rem; font-weight: 650; display: block; margin-bottom: 4px; color: var(--text); }
.panel-desc { font-size: 0.8rem; color: var(--muted); margin: 0 0 10px; }
.stack { display: flex; flex-direction: column; gap: 14px; }
.row { display: flex; gap: 8px; flex-wrap: wrap; align-items: center; }
.row-between { display: flex; gap: 10px; flex-wrap: wrap; align-items: center; justify-content: space-between; }
.divider { border: 0; border-top: 1px solid var(--border); margin: 16px 0; }

.grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
.grid-2 > *, .grid-2-60 > *, .grid-3 > * { min-width: 0; }
.grid-3 { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 14px; }
.grid-2-60 { display: grid; grid-template-columns: 1.15fr 1fr; gap: 16px; align-items: start; }
@media (max-width: 900px) {
  .grid-2, .grid-2-60 { grid-template-columns: 1fr; }
}

/* ---------- Plegables (opciones avanzadas) ---------- */
details.plegable {
  border: 1px solid var(--border);
  border-radius: 10px;
  background: var(--surface-2);
  margin: 12px 0;
}
details.plegable > summary {
  list-style: none;
  cursor: pointer;
  padding: 10px 14px;
  font-size: 0.84rem;
  font-weight: 600;
  color: var(--text-2);
  display: flex; align-items: center; gap: 8px;
  user-select: none;
}
details.plegable > summary::-webkit-details-marker { display: none; }
details.plegable > summary::before {
  content: ""; width: 7px; height: 7px;
  border-right: 2px solid var(--muted); border-bottom: 2px solid var(--muted);
  transform: rotate(-45deg); transition: transform 0.15s; margin-right: 2px;
}
details.plegable[open] > summary::before { transform: rotate(45deg); }
details.plegable > summary .hint { font-weight: 400; color: var(--muted); font-size: 0.8rem; }
details.plegable > .plegable-body { padding: 2px 14px 14px; }

/* ---------- Formularios ---------- */
.campo label, label.lbl {
  display: block;
  font-size: 0.8rem;
  font-weight: 600;
  color: var(--text-2);
  margin-bottom: 5px;
}
input[type="text"], input[type="number"], select, textarea {
  width: 100%;
  font-size: 0.88rem;
  padding: 8px 10px;
  border: 1px solid var(--border-strong);
  border-radius: var(--radius-sm);
  background: #fff;
  color: var(--text);
  font-family: inherit;
  transition: border-color 0.12s, box-shadow 0.12s;
}
input[type="text"]:focus, input[type="number"]:focus, select:focus, textarea:focus {
  outline: none;
  border-color: var(--primary);
  box-shadow: 0 0 0 3px rgba(37,99,235,0.14);
}
input.tiempo { font-family: var(--mono); font-weight: 600; letter-spacing: 0.02em; }
input[type="checkbox"] { accent-color: var(--primary); width: 15px; height: 15px; }
label.check {
  display: flex; align-items: flex-start; gap: 8px;
  font-size: 0.84rem; color: var(--text-2); cursor: pointer; font-weight: 400;
}
label.check input { margin-top: 2px; flex-shrink: 0; }
.input-con-btn { display: flex; gap: 6px; }
.input-con-btn input { flex: 1; min-width: 0; }

/* ---------- Botones ---------- */
button, .btn-link {
  font-family: inherit;
  font-size: 0.86rem;
  font-weight: 600;
  padding: 8px 14px;
  border-radius: var(--radius-sm);
  border: 1px solid transparent;
  cursor: pointer;
  display: inline-flex;
  align-items: center;
  gap: 6px;
  transition: background 0.12s, border-color 0.12s, box-shadow 0.12s;
  line-height: 1.3;
}
button:focus-visible { outline: 2px solid var(--primary); outline-offset: 2px; }
button:disabled { opacity: 0.45; cursor: not-allowed; }
.btn-primary { background: var(--primary); color: #fff; box-shadow: 0 1px 2px rgba(37,99,235,0.25); }
.btn-primary:hover:not(:disabled) { background: var(--primary-hover); }
.btn-secondary { background: var(--secondary); color: #fff; }
.btn-secondary:hover:not(:disabled) { background: var(--secondary-hover); }
.btn-outline { background: #fff; color: var(--text-2); border-color: var(--border-strong); }
.btn-outline:hover:not(:disabled) { background: #f3f4f6; border-color: #c4c8cf; }
.btn-ghost { background: transparent; color: var(--primary); border-color: transparent; }
.btn-ghost:hover:not(:disabled) { background: var(--primary-light); }
.btn-gold { background: var(--warn-bg); color: #b45309; border-color: #fde68a; }
.btn-gold.activo { background: var(--ok); color: #fff; border-color: var(--ok); }
.btn-danger-soft { background: var(--danger-bg); color: var(--danger); border-color: #fecaca; }
.btn-danger-soft:hover:not(:disabled) { background: #fee2e2; }
.btn-sm { padding: 6px 11px; font-size: 0.8rem; }
.btn-xs { padding: 4px 9px; font-size: 0.75rem; border-radius: 6px; }
.btn-block { width: 100%; justify-content: center; }
.btn-lg { padding: 11px 16px; font-size: 0.92rem; }

/* ---------- Control segmentado ---------- */
.segmentado { display: grid; grid-template-columns: 1fr 1fr; gap: 4px; padding: 4px; background: #eef0f3; border-radius: 10px; }
.segmentado label { margin: 0; cursor: pointer; }
.segmentado input { position: absolute; opacity: 0; pointer-events: none; }
.segmentado span {
  display: block; text-align: center; padding: 7px 10px; border-radius: 7px;
  font-size: 0.85rem; font-weight: 600; color: var(--muted); transition: background 0.12s, color 0.12s;
}
.segmentado input:checked + span { background: #fff; color: var(--text); box-shadow: 0 1px 3px rgba(17,24,39,0.12); }
.segmentado input:focus-visible + span { outline: 2px solid var(--primary); }
.partes-lista { display: flex; flex-direction: column; gap: 6px; margin-top: 10px; max-height: 360px; overflow-y: auto; }
.parte-item {
  display: grid; grid-template-columns: 34px 1fr auto; gap: 10px; align-items: center;
  padding: 7px 10px; border: 1px solid var(--border); border-radius: 8px; background: var(--surface-2); font-size: 0.84rem;
}
.parte-item .pn { font-weight: 700; color: var(--primary); font-family: var(--mono); }
.parte-item .pt { font-family: var(--mono); font-size: 0.8rem; color: var(--text-2); }
.parte-item a { text-decoration: none; }

/* ---------- Video para publicar ---------- */
.voz-fila {
  display: grid; grid-template-columns: 14px minmax(140px, 1.2fr) 1fr auto auto; gap: 12px; align-items: center;
  padding: 10px 12px; border: 1px solid var(--border); border-radius: 10px; background: #fff;
  transition: border-color 0.12s, background 0.12s;
}
.voz-fila.silenciada { border-color: #fecaca; background: #fffafa; }
.voz-fila .dot { width: 12px; height: 12px; border-radius: 50%; }
.voz-fila input[type="text"] { padding: 6px 8px; font-weight: 600; }
.voz-fila .voz-stats { font-size: 0.78rem; color: var(--muted); }
.voz-fila .voz-stats .etq { color: var(--danger); font-weight: 600; }
@media (max-width: 760px) { .voz-fila { grid-template-columns: 14px 1fr auto; } .voz-fila .voz-stats { grid-column: 2 / -1; } }
.switch { display: inline-flex; align-items: center; gap: 8px; cursor: pointer; font-size: 0.82rem; font-weight: 600; color: var(--text-2); user-select: none; }
.switch input { position: absolute; opacity: 0; pointer-events: none; }
.switch .pista { width: 36px; height: 20px; border-radius: 999px; background: #d1d5db; position: relative; transition: background 0.15s; flex-shrink: 0; }
.switch .pista::after { content: ""; position: absolute; top: 2px; left: 2px; width: 16px; height: 16px; border-radius: 50%; background: #fff; transition: transform 0.15s; box-shadow: 0 1px 2px rgba(0,0,0,0.2); }
.switch input:checked + .pista { background: var(--danger); }
.switch input:checked + .pista::after { transform: translateX(16px); }
.switch input:focus-visible + .pista { outline: 2px solid var(--primary); outline-offset: 2px; }
.pub-resultado video { width: 100%; max-height: 420px; background: #000; border-radius: 10px; }

/* Enlaces con aspecto de botón */
a[class*="btn-"] {
  display: inline-flex; align-items: center; gap: 6px; text-decoration: none; cursor: pointer;
  border: 1px solid transparent; border-radius: var(--radius-sm);
  padding: 8px 14px; font-size: 0.86rem; font-weight: 600; line-height: 1.3;
}
a.btn-outline { border-color: var(--border-strong); }
a.btn-sm { padding: 6px 11px; font-size: 0.8rem; }
a.btn-xs { padding: 4px 9px; font-size: 0.75rem; border-radius: 6px; }

.seg3 { display: inline-grid; grid-template-columns: repeat(3, auto); gap: 2px; padding: 3px; background: #eef0f3; border-radius: 8px; }
.seg3 label { margin: 0; cursor: pointer; }
.seg3 input { position: absolute; opacity: 0; pointer-events: none; }
.seg3 span { display: block; padding: 5px 9px; border-radius: 6px; font-size: 0.76rem; font-weight: 600; color: var(--muted); white-space: nowrap; }
.seg3 input:checked + span { background: #fff; color: var(--text); box-shadow: 0 1px 2px rgba(17,24,39,0.12); }
.seg3 input[value="silenciar"]:checked + span { background: var(--danger); color: #fff; }
.seg3 input[value="subtitulo"]:checked + span { background: var(--primary); color: #fff; }
.seg3 input:focus-visible + span { outline: 2px solid var(--primary); }
.voz-fila.subtitulo { border-color: var(--primary-border); background: #fafbff; }
.voz-fila .voz-stats .etq.azul { color: var(--primary); }

/* ---------- Vista previa de Video para publicar ---------- */
.pub-dos-col { display: grid; grid-template-columns: minmax(0, 1.35fr) minmax(300px, 1fr); gap: 16px; align-items: start; }
.pub-prev-col { position: sticky; top: 12px; }
.pub-prev-col .preview { margin-top: 0; }
@media (max-width: 1000px) { .pub-dos-col { grid-template-columns: 1fr; } .pub-prev-col { position: static; order: -1; } }
.pub-dos-col .voz-fila { grid-template-columns: 12px minmax(120px, 1fr) auto; }
.pub-dos-col .voz-fila .voz-stats { grid-column: 2 / -1; order: 5; }
.pub-dos-col .voz-fila .seg3 { grid-column: 2 / -1; order: 6; justify-self: start; }
.pub-video-wrap { position: relative; }
.pub-video-wrap video { width: 100%; max-height: 340px; background: #000; border-radius: 8px; display: block; }
.pub-overlay { bottom: 44px; }
.pub-overlay span { font-size: 0.92rem; }
.pub-mute-tag { position: absolute; top: 8px; left: 8px; background: var(--danger); color: #fff; font-size: 0.74rem; font-weight: 700; padding: 3px 8px; border-radius: 999px; }
.tl-seg { position: absolute; top: 0; bottom: 0; min-width: 1px; }
.tl-seg.sil { background: rgba(220,38,38,0.55); }
.tl-seg.sub { background: rgba(37,99,235,0.55); }
.tl-leyenda { display: flex; gap: 14px; font-size: 0.74rem; color: var(--muted); margin-top: 4px; }
.tl-leyenda i { display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 4px; vertical-align: -1px; }
.pub-ahora { font-size: 0.82rem; color: var(--text-2); margin-top: 8px; min-height: 1.3em; }

/* ---------- Vincular (paso 2) y Resultados (paso 4) ---------- */
#p4-video { width: 100%; max-height: 340px; background: #000; border-radius: 8px; display: block; }
.vacio-mini { font-size: 0.82rem; color: var(--muted); border: 1px dashed var(--border-strong); border-radius: 8px; padding: 14px; text-align: center; margin-top: 8px; }
.res-top { display: grid; grid-template-columns: minmax(0, 1fr) minmax(320px, 0.9fr); gap: 18px; align-items: start; }
@media (max-width: 1000px) { .res-top { grid-template-columns: 1fr; } }
.res-material { font-size: 0.86rem; color: var(--text-2); margin-top: 6px; display: flex; flex-direction: column; gap: 4px; }
.res-material .fila { display: flex; gap: 8px; align-items: baseline; flex-wrap: wrap; }
.res-material .k { color: var(--muted); min-width: 92px; font-size: 0.8rem; }
.res-material .ok { color: var(--ok); font-weight: 600; }
.res-material .pend { color: var(--warn); font-weight: 600; }
.res-grupo-tit { font-size: 0.74rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em; color: var(--muted); margin: 22px 0 8px; }
.voces-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(330px, 1fr)); gap: 8px; }
.voces-grid .voz-fila { grid-template-columns: 12px minmax(0, 1fr) auto; }
.voces-grid .voz-fila > .dot { grid-column: 1; grid-row: 1; }
.voces-grid .voz-fila > .voz-nom { grid-column: 2; grid-row: 1; }
.voces-grid .voz-fila > .row { grid-column: 3; grid-row: 1; }
.voces-grid .voz-fila > .voz-stats { grid-column: 2 / -1; grid-row: 2; }
.voces-grid .voz-fila > .seg3 { grid-column: 2 / -1; grid-row: 3; justify-self: start; }
.voz-fila .voz-nom { font-weight: 600; font-size: 0.88rem; overflow-wrap: anywhere; }

#res-video-card.deshabilitada #btn-pub-generar { opacity: 0.45; pointer-events: none; }

.ses-item { display: flex; justify-content: space-between; align-items: center; gap: 12px; padding: 10px 12px; border: 1px solid var(--border); border-radius: 10px; background: #fff; }
@media (max-width: 760px) { .ses-item { flex-direction: column; align-items: flex-start; } }

/* ---------- Recetas ---------- */
.recetas { display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; margin: 0 0 14px; }
@media (max-width: 1000px) { .recetas { grid-template-columns: 1fr 1fr; } }
.recetas-tit { grid-column: 1 / -1; font-size: 0.72rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em; color: var(--primary); }
.receta {
  display: flex; flex-direction: column; align-items: flex-start; gap: 2px; text-align: left;
  background: #fff; border: 1px solid var(--border); border-radius: 10px; padding: 10px 12px; font-weight: 400;
}
.receta strong { font-size: 0.84rem; color: var(--text); }
.receta span { font-size: 0.76rem; color: var(--muted); }
.receta:hover { border-color: var(--primary-border); background: #fafbff; }
.receta.activa { border-color: var(--primary); box-shadow: 0 0 0 3px rgba(37,99,235,0.14); background: var(--primary-light); }
.btn-yo { background: #fff; border: 1px solid var(--border-strong); color: var(--text-2); padding: 4px 9px; font-size: 0.75rem; border-radius: 6px; white-space: nowrap; }
.btn-yo.activo { background: #fef3c7; border-color: #f59e0b; color: #92400e; }

/* ---------- Hablantes (paso 3) ---------- */
.ctx-bar { display: flex; justify-content: space-between; align-items: center; gap: 12px; flex-wrap: wrap; padding: 14px 18px; }
.ctx-tit { display: flex; align-items: baseline; gap: 8px; flex-wrap: wrap; }
.ctx-label { font-size: 0.72rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em; color: var(--muted); }
.ctx-tit strong { font-size: 0.98rem; overflow-wrap: anywhere; }
.ctx-datos { font-size: 0.82rem; color: var(--text-2); margin-top: 2px; }
.ctx-datos .ok { color: var(--ok); font-weight: 600; }
.ctx-datos .pend { color: var(--warn); font-weight: 600; }
.paso-cab { display: flex; gap: 12px; align-items: flex-start; margin-bottom: 14px; }
.paso-cab h2 { margin: 0 0 2px; }
.paso-cab p { margin: 0; font-size: 0.86rem; color: var(--muted); }
.paso-n {
  flex-shrink: 0; width: 30px; height: 30px; border-radius: 9px; background: var(--primary); color: #fff;
  display: grid; place-items: center; font-weight: 700; font-size: 0.9rem;
}
.voz-row {
  display: grid; grid-template-columns: 12px minmax(180px, 1.3fr) minmax(140px, 1fr) auto auto; gap: 12px; align-items: center;
  padding: 8px 12px; border: 1px solid var(--border); border-radius: 10px; background: #fff;
}
.voz-row.es-yo { border-color: #f59e0b; background: #fffdf5; }
.voz-row .dot { width: 12px; height: 12px; border-radius: 50%; }
.voz-row input[type="text"] { padding: 6px 8px; font-weight: 600; }
.voz-ej { font-size: 0.75rem; color: var(--muted); font-style: italic; margin-top: 3px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: 100%; }
.voz-stats { font-size: 0.78rem; color: var(--muted); }
.voz-stats .guardada { color: var(--ok); }
.voz-stats .cambio { color: var(--primary); font-weight: 600; }
.voz-row .voz-nombre { min-width: 0; }
@media (max-width: 760px) {
  .voz-row { grid-template-columns: 12px minmax(0, 1fr) auto; }
  .voz-row .voz-stats { grid-column: 2 / 3; }
}
.salidas-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 12px; align-items: start; }
.salida { border: 1px solid var(--border); border-radius: 12px; padding: 14px; background: #fff; display: flex; flex-direction: column; gap: 10px; }
.salida-tit { font-weight: 650; font-size: 0.95rem; }
.salida-desc { margin: -6px 0 0; font-size: 0.8rem; color: var(--muted); }
.salida-resumen { font-size: 0.78rem; color: #1e40af; background: var(--primary-light); border-radius: 8px; padding: 6px 8px; }
.salida-resumen:empty { display: none; }
.aviso-mini { font-size: 0.8rem; color: #92400e; background: #fffbeb; border: 1px solid #fde68a; border-radius: 8px; padding: 6px 8px; }
.aviso-mini:empty { display: none; }
.salida-ok { margin-top: 4px; padding: 10px; border-radius: 8px; background: #f0fdf4; border: 1px solid #bbf7d0; }
.voces-check { display: flex; flex-direction: column; gap: 4px; max-height: 170px; overflow-y: auto; padding: 8px; border: 1px solid var(--border); border-radius: 8px; background: var(--surface-2); }
.voces-check .check { font-size: 0.82rem; }
.dot-mini { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 6px; vertical-align: 1px; }

/* ---------- Badges ---------- */
.badge {
  font-size: 0.73rem;
  padding: 3px 9px;
  border-radius: 999px;
  background: #eef0f3;
  color: var(--text-2);
  font-weight: 600;
  white-space: nowrap;
  display: inline-block;
}
.badge-azul { background: #e0e9ff; color: #1e40af; }
.badge-verde { background: #dcfce7; color: #15803d; }
.badge-ambar { background: #fef3c7; color: #b45309; }

/* ---------- Zona de archivos ---------- */
.dropzone {
  border: 1.5px dashed #b8bfca;
  border-radius: var(--radius);
  padding: 28px 18px;
  text-align: center;
  background: var(--surface-2);
  cursor: pointer;
  transition: border-color 0.15s, background 0.15s;
  display: flex; flex-direction: column; align-items: center; justify-content: center;
  gap: 4px;
}
.dropzone:hover, .dropzone.sobre { border-color: var(--primary); background: var(--primary-light); }
.dropzone .dz-icon {
  width: 44px; height: 44px; border-radius: 12px;
  background: #fff; border: 1px solid var(--border);
  display: grid; place-items: center; color: var(--primary); margin-bottom: 6px;
}
.dropzone .dz-icon svg { width: 22px; height: 22px; }
.dropzone .dz-title { font-weight: 600; font-size: 0.95rem; color: var(--text); }
.dropzone .dz-sub { font-size: 0.8rem; color: var(--muted); }

.dropzone.dz-compacta { flex-direction: row; justify-content: flex-start; gap: 14px; padding: 12px 14px; }
.dropzone.dz-compacta .dz-icon { margin: 0; width: 38px; height: 38px; flex-shrink: 0; }

/* ---------- Vista previa (paso 1) ---------- */
.dropzone.con-archivo { min-height: 0 !important; flex-direction: row; justify-content: flex-start; gap: 12px; padding: 12px 14px; text-align: left; }
.dropzone.con-archivo .dz-icon { margin: 0; width: 36px; height: 36px; flex-shrink: 0; }
.preview { margin-top: 12px; border: 1px solid var(--border); border-radius: 12px; padding: 12px; background: var(--surface-2); }
.preview video { width: 100%; max-height: 300px; background: #000; border-radius: 8px; display: block; }
.prev-reloj { display: flex; align-items: baseline; gap: 8px; flex-wrap: wrap; margin: 10px 0 6px; }
.t-grande { font-family: var(--mono); font-size: 1.7rem; font-weight: 600; color: var(--text); letter-spacing: 0.02em; }
.t-dur { font-family: var(--mono); font-size: 0.9rem; color: var(--muted); }
.prev-saltos { margin-left: auto; display: inline-flex; gap: 4px; }
.timeline { position: relative; height: 34px; background: #e9ecf1; border-radius: 6px; cursor: pointer; overflow: hidden; }
.tl-capa { position: absolute; inset: 0; }
.tl-rango { position: absolute; top: 0; bottom: 0; background: rgba(37,99,235,0.22); border-left: 2px solid var(--primary); border-right: 2px solid var(--primary); }
.tl-parte { position: absolute; top: 0; bottom: 0; display: grid; place-items: center; font-size: 0.68rem; font-weight: 700; color: var(--primary-hover); font-family: var(--mono); overflow: hidden; }
.tl-parte.par { background: rgba(37,99,235,0.18); }
.tl-parte.impar { background: rgba(37,99,235,0.08); }
.tl-corte { position: absolute; top: 0; bottom: 0; width: 2px; background: var(--danger); }
.tl-cabezal { position: absolute; top: -2px; bottom: -2px; width: 2px; background: #111827; pointer-events: none; left: 0; }
.tl-cabezal::before { content: ""; position: absolute; top: 0; left: -4px; border: 5px solid transparent; border-top-color: #111827; }

/* ---------- Cola y progreso ---------- */
.cola-lista { list-style: none; padding: 0; margin: 10px 0 0; display: flex; flex-direction: column; gap: 6px; }
.cola-item {
  display: flex; align-items: center; justify-content: space-between;
  padding: 8px 12px; background: var(--surface-2);
  border: 1px solid var(--border); border-radius: 8px; font-size: 0.86rem;
}
.progreso-item {
  border: 1px solid var(--border); border-radius: 10px;
  padding: 12px 14px; margin-top: 12px; background: #fff;
}
.barra-fondo { height: 7px; background: #e9ecf1; border-radius: 999px; overflow: hidden; margin-top: 6px; }
.barra-relleno { height: 100%; background: var(--primary); width: 0%; transition: width 0.25s; }

/* ---------- Paso 3 ---------- */
.grupo-bar {
  display: flex; justify-content: space-between; align-items: center;
  gap: 12px; flex-wrap: wrap;
}
.grupo-bar select { width: auto; min-width: 200px; font-weight: 600; }
.diagnostico-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; margin-bottom: 16px; }
@media (max-width: 900px) { .diagnostico-grid { grid-template-columns: 1fr; } }
.diag-box {
  border: 1px solid var(--border);
  border-left: 4px solid var(--border-strong);
  border-radius: 10px;
  padding: 14px 16px;
  background: #fff;
}
.diag-box.ok { border-left-color: var(--ok); background: #fbfefc; }
.diag-box.warn { border-left-color: var(--warn); background: #fffdf7; }
.diag-box h4 { margin: 0 0 6px; font-size: 0.88rem; font-weight: 650; }
.diag-box p { margin: 0; font-size: 0.85rem; color: var(--text-2); }

.player-sticky {
  position: sticky; top: 10px; z-index: 25;
  background: #111827; color: #f9fafb;
  padding: 10px 14px; border-radius: 12px; margin-bottom: 16px;
  display: flex; flex-wrap: wrap; align-items: center; gap: 12px;
  box-shadow: 0 8px 24px rgba(17,24,39,0.22);
}
.player-sticky audio { flex: 1; min-width: 240px; height: 36px; }

.hablantes-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 14px; }
.spk-card {
  border: 1px solid var(--border); border-radius: 12px; padding: 14px;
  background: #fff; display: flex; flex-direction: column; gap: 10px;
  box-shadow: var(--shadow-sm);
}
.spk-card.es-mi-voz { border-color: #86efac; box-shadow: 0 0 0 3px rgba(22,163,74,0.12); background: #fbfefc; }
.spk-card.silenciado { border-color: #fecaca; background: #fffafa; opacity: 0.8; }
.spk-frases-preview {
  font-size: 0.8rem; color: #4b5563; background: var(--surface-2);
  border-left: 3px solid var(--border-strong); padding: 6px 9px; border-radius: 4px; font-style: italic;
}

.segmentos-wrap {
  max-height: 460px; overflow-y: auto;
  border: 1px solid var(--border); border-radius: 10px;
  background: var(--surface-2); padding: 8px;
}
.seg-row {
  display: grid; grid-template-columns: auto 155px 1fr auto; gap: 8px; align-items: start;
  padding: 8px 10px; border-radius: 8px; background: #fff;
  border: 1px solid #eceef1; margin-bottom: 6px;
}
@media (max-width: 720px) { .seg-row { grid-template-columns: 1fr; } }
.seg-row.seg-silenciado { opacity: 0.45; background: #fffafa; }
.seg-row.seg-silenciado .seg-texto { text-decoration: line-through; }
.seg-row.fuera-de-rango { opacity: 0.35; background: #f3f4f6; }
.seg-row.marca-inicio { border-left: 4px solid var(--ok); }
.seg-row.marca-fin { border-right: 4px solid var(--danger); }
.seg-time-btn {
  font-family: var(--mono); font-size: 0.76rem; padding: 4px 8px;
  background: #f3f4f6; color: var(--text-2); border: 1px solid var(--border-strong); border-radius: 6px;
}

/* ---------- Paso 4 ---------- */
.vinculo-bar {
  display: grid; grid-template-columns: 1fr auto 1fr; gap: 14px; align-items: end;
}
@media (max-width: 900px) { .vinculo-bar { grid-template-columns: 1fr; } .vinculo-bar .vinculo-flecha { display: none; } }
.vinculo-flecha { color: var(--muted); padding-bottom: 8px; }
.vinculo-flecha svg { width: 20px; height: 20px; }
.video-col { position: sticky; top: 16px; }
.video-stage {
  position: relative; background: #000; border-radius: 12px; overflow: hidden;
  margin: 0 0 8px; aspect-ratio: 16 / 9; display: flex; align-items: center; justify-content: center;
}
.video-stage video { width: 100%; height: 100%; }
.sub-overlay { position: absolute; bottom: 48px; left: 6%; right: 6%; text-align: center; pointer-events: none; z-index: 10; }
.sub-overlay span {
  display: inline-block; background: rgba(0,0,0,0.82); color: #fff;
  font-size: 1.05rem; font-weight: 600; padding: 6px 14px; border-radius: 6px;
  line-height: 1.35; text-shadow: 0 1px 2px #000;
}
.paso-num {
  display: inline-grid; place-items: center; width: 22px; height: 22px; border-radius: 6px;
  background: var(--primary-light); color: var(--primary); font-size: 0.75rem; font-weight: 700;
  margin-right: 6px; vertical-align: 1px;
}

/* ---------- Paso 5 ---------- */
.metricas { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; margin-bottom: 14px; }
.metrica { background: var(--surface-2); border: 1px solid var(--border); border-radius: 10px; padding: 10px; text-align: center; }
.metrica .valor { font-size: 1.25rem; font-weight: 700; color: var(--text); }
.metrica .lbl { font-size: 0.72rem; color: var(--muted); }
.chips { display: flex; flex-wrap: wrap; gap: 5px; }

/* ---------- Estado vacío ---------- */
.vacio {
  text-align: center; padding: 56px 24px;
  border-style: dashed; background: transparent; box-shadow: none;
}
.vacio .vacio-icon {
  width: 52px; height: 52px; margin: 0 auto 12px; border-radius: 14px;
  background: #fff; border: 1px solid var(--border); display: grid; place-items: center; color: var(--muted);
}
.vacio .vacio-icon svg { width: 24px; height: 24px; }
.vacio h3 { font-size: 1.05rem; margin-bottom: 4px; }
.vacio p { color: var(--muted); font-size: 0.9rem; margin: 0 auto 16px; max-width: 460px; }

/* ---------- Ayuda (modal) ---------- */
.modal {
  position: fixed; inset: 0; z-index: 100;
  background: rgba(17,24,39,0.45);
  display: none; align-items: flex-start; justify-content: center;
  padding: 5vh 16px; overflow-y: auto;
}
.modal.abierto { display: flex; }
.modal-box {
  background: #fff; border-radius: 16px; max-width: 860px; width: 100%;
  box-shadow: 0 24px 60px rgba(17,24,39,0.28); padding: 26px 28px;
  animation: aparecer 0.18s ease;
}
.modal-head { display: flex; justify-content: space-between; align-items: flex-start; gap: 12px; margin-bottom: 16px; }
.modal-head h2 { margin: 0; font-size: 1.25rem; }
.modal-head p { margin: 4px 0 0; color: var(--muted); font-size: 0.9rem; }
.btn-cerrar { background: transparent; border: 0; color: var(--muted); padding: 6px; border-radius: 8px; }
.btn-cerrar:hover { background: #f3f4f6; color: var(--text); }
.btn-cerrar svg { width: 18px; height: 18px; }
.guia-lista { display: flex; flex-direction: column; gap: 8px; }
.guia-item {
  all: unset; box-sizing: border-box; cursor: pointer;
  display: grid; grid-template-columns: 32px 1fr auto; gap: 12px; align-items: center;
  padding: 12px 14px; border: 1px solid var(--border); border-radius: 10px;
  transition: border-color 0.12s, background 0.12s;
}
.guia-item:hover { border-color: var(--primary-border); background: #fafbff; }
.guia-item .gnum {
  width: 30px; height: 30px; border-radius: 8px; background: var(--primary-light);
  color: var(--primary); display: grid; place-items: center; font-weight: 700; font-size: 0.85rem;
}
.guia-item strong { display: block; font-size: 0.92rem; }
.guia-item span { font-size: 0.82rem; color: var(--muted); }
.guia-item .ir { color: var(--muted); font-size: 0.8rem; white-space: nowrap; }
.guia-rama { margin-left: 22px; padding-left: 20px; border-left: 2px solid var(--border); display: flex; flex-direction: column; gap: 8px; }
.guia-nota {
  margin-top: 16px; font-size: 0.82rem; color: var(--text-2);
  background: var(--surface-2); border: 1px solid var(--border); border-radius: 10px; padding: 12px 14px;
}

input.tiempo::placeholder { font-family: var(--font); font-weight: 400; letter-spacing: 0; }
#p3-resumen-audio:empty, #p4-preview-sincro:empty, #p4-match-diagnostico:empty { display: none; }
.badge { max-width: 100%; white-space: normal; }
.app { background: linear-gradient(to right, #fbfbfc var(--sidebar-w), var(--bg) var(--sidebar-w)); }

/* ---------- Responsive: barra lateral a barra superior ---------- */
@media (max-width: 860px) {
  .app { grid-template-columns: 1fr; }
  .sidebar {
    position: sticky; top: 0; z-index: 50; height: auto;
    flex-direction: row; align-items: center; gap: 10px;
    padding: 10px 12px; border-right: 0; border-bottom: 1px solid var(--border);
    overflow-x: auto; overflow-y: hidden;
  }
  .brand-name, .brand-sub, .nav-group-label, .step-btn .step-sub, #git-bar { display: none; }
  .brand { padding: 0; }
  .sidebar nav { display: contents; }
  .pasos-stepper { flex-direction: row; }
  .step-btn { grid-template-columns: 24px auto; white-space: nowrap; padding: 7px 10px; }
  .step-btn.activo::before { display: none; }
  .sidebar-foot { margin: 0; flex-direction: row; }
  .side-link span { display: none; }
  .main { padding: 20px 16px 60px; }
  .app { background: var(--bg); }
}
</style>
</head>
<body>

<!-- Íconos reutilizables -->
<svg width="0" height="0" style="position:absolute" aria-hidden="true">
  <defs>
    <symbol id="i-wave" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M2 10v3"/><path d="M6 6v11"/><path d="M10 3v18"/><path d="M14 8v7"/><path d="M18 5v13"/><path d="M22 10v3"/></symbol>
    <symbol id="i-film" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="18" height="18" rx="2"/><path d="M7 3v18M17 3v18M3 7.5h4M3 12h18M3 16.5h4M17 7.5h4M17 16.5h4"/></symbol>
    <symbol id="i-text" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 3H6a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9z"/><path d="M14 3v6h6M8 13h8M8 17h5"/></symbol>
    <symbol id="i-users" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M22 21v-2a4 4 0 0 0-3-3.87M16 3.13a4 4 0 0 1 0 7.75"/></symbol>
    <symbol id="i-cc" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="2" y="5" width="20" height="14" rx="2"/><path d="M10 10.5a2 2 0 1 0 0 3M17 10.5a2 2 0 1 0 0 3"/></symbol>
    <symbol id="i-spark" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3l1.9 5.8L20 11l-6.1 2.2L12 19l-1.9-5.8L4 11l6.1-2.2z"/></symbol>
    <symbol id="i-help" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><path d="M9.1 9a3 3 0 0 1 5.8 1c0 2-3 3-3 3M12 17h.01"/></symbol>
    <symbol id="i-git" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="6" cy="6" r="3"/><circle cx="6" cy="18" r="3"/><circle cx="18" cy="9" r="3"/><path d="M6 9v6M18 12c0 3-3 4-9 5"/></symbol>
    <symbol id="i-upload" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4M17 8l-5-5-5 5M12 3v12"/></symbol>
    <symbol id="i-x" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M18 6 6 18M6 6l12 12"/></symbol>
    <symbol id="i-mute" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M11 5 6 9H2v6h4l5 4z"/><path d="m22 9-6 6M16 9l6 6"/></symbol>
    <symbol id="i-link" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M10 13a5 5 0 0 0 7.5.5l3-3a5 5 0 0 0-7-7l-1.7 1.7M14 11a5 5 0 0 0-7.5-.5l-3 3a5 5 0 0 0 7 7l1.7-1.7"/></symbol>
  </defs>
</svg>

<div class="app">

<!-- ====================================================================== -->
<!-- BARRA LATERAL                                                          -->
<!-- ====================================================================== -->
<aside class="sidebar">
  <div class="brand">
    <div class="brand-logo"><svg><use href="#i-wave"/></svg></div>
    <div>
      <div class="brand-name">Estudio de Audio</div>
      <div class="brand-sub">100% local en tu PC</div>
    </div>
  </div>

  <nav class="pasos-stepper" id="stepper-nav" aria-label="Secciones">
    <div class="nav-group-label">Flujo de trabajo</div>
    <button type="button" class="step-btn activo" data-step="1">
      <span class="step-num">1</span>
      <span class="step-title">Video / Audio</span>
      <span class="step-sub">Extraer, recortar, dividir</span>
    </button>
    <button type="button" class="step-btn" data-step="2">
      <span class="step-num">2</span>
      <span class="step-title">Transcripción</span>
      <span class="step-sub">Crear, vincular, sincronizar</span>
    </button>
    <button type="button" class="step-btn" data-step="3">
      <span class="step-num">3</span>
      <span class="step-title">Hablantes</span>
      <span class="step-sub">Quién es quién y revisar</span>
    </button>
    <button type="button" class="step-btn" data-step="4">
      <span class="step-num">4</span>
      <span class="step-title">Resultados</span>
      <span class="step-sub">Audio, texto, subtítulos, video</span>
    </button>
    <div class="nav-group-label" style="margin-top:14px">Herramientas</div>
    <button type="button" class="step-btn" data-step="5">
      <span class="step-num"><svg><use href="#i-spark"/></svg></span>
      <span class="step-title">Mi estilo y guiones</span>
      <span class="step-sub">Entrenar y convertir textos</span>
    </button>
  </nav>

  <div class="sidebar-foot">
    <button type="button" class="side-link" id="btn-toggle-guia" onclick="toggleGuiaBienvenida()">
      <svg><use href="#i-help"/></svg><span>Cómo funciona</span>
    </button>
    <div id="git-bar">
      <div class="git-head"><svg><use href="#i-git"/></svg> GitHub</div>
      <strong id="git-estado-txt">Verificando…</strong>
      <div class="git-acciones">
        <button type="button" class="btn-outline btn-xs" id="btn-git-sinc">Sincronizar</button>
        <button type="button" class="btn-outline btn-xs" id="btn-git-cfg" style="display:none">Configurar</button>
      </div>
    </div>
  </div>
</aside>

<!-- ====================================================================== -->
<!-- AYUDA: CÓMO FUNCIONA (se abre sola la primera vez)                      -->
<!-- ====================================================================== -->
<div class="modal" id="card-bienvenida-guia" role="dialog" aria-modal="true" aria-labelledby="guia-titulo" onclick="if(event.target===this) toggleGuiaBienvenida(false)">
  <div class="modal-box">
    <div class="modal-head">
      <div>
        <h2 id="guia-titulo">Cómo funciona</h2>
        <p>Todo se procesa en tu PC. No hace falta empezar por el paso 1: entrá directo a la sección que necesites según el archivo que tengas.</p>
      </div>
      <button type="button" class="btn-cerrar" onclick="toggleGuiaBienvenida(false)" aria-label="Cerrar"><svg><use href="#i-x"/></svg></button>
    </div>

    <div class="guia-lista">
      <button type="button" class="guia-item" onclick="toggleGuiaBienvenida(false); irAPaso(1)">
        <span class="gnum">1</span>
        <div><strong>Video / Audio</strong><span>Subís un video o audio y obtenés el MP3, recortás un tramo por minuto:segundo o lo dividís en partes (en varios puntos o cada X minutos).</span></div>
        <span class="ir">Ir →</span>
      </button>
      <button type="button" class="guia-item" onclick="toggleGuiaBienvenida(false); irAPaso(2)">
        <span class="gnum">2</span>
        <div><strong>Transcripción</strong><span>Crearla con Whisper o subir una ya hecha (Tactiq, YouTube, SRT…), <strong>vincularla con su video</strong> y sincronizarla. Con Whisper queda sincronizada sola.</span></div>
        <span class="ir">Ir →</span>
      </button>
      <button type="button" class="guia-item" onclick="toggleGuiaBienvenida(false); irAPaso(3)">
        <span class="gnum">3</span>
        <div><strong>Hablantes</strong><span>Quién es quién (a mano, con Tactiq, por presentaciones o con una lista), cuál sos vos, y revisar línea por línea. Las correcciones quedan guardadas.</span></div>
        <span class="ir">Ir →</span>
      </button>
      <div class="guia-rama">
        <button type="button" class="guia-item" onclick="toggleGuiaBienvenida(false); irAPaso(4)">
          <span class="gnum">4</span>
          <div><strong>Resultados</strong><span>Con todo conectado: tu voz limpia, audio sin ciertas voces, transcripción, subtítulos SRT/VTT y video con voces silenciadas o subtituladas (recetas), con vista previa.</span></div>
          <span class="ir">Ir →</span>
        </button>
        <button type="button" class="guia-item" onclick="toggleGuiaBienvenida(false); irAPaso(5)">
          <span class="gnum"><svg width="15" height="15"><use href="#i-spark"/></svg></span>
          <div><strong>Mi estilo y guiones</strong><span>Entrena un modelo con tus transcripciones (muletillas, conectores, ritmo) y reescribe cualquier guión con tu forma de hablar.</span></div>
          <span class="ir">Ir →</span>
        </button>
      </div>
    </div>

    <div class="ejemplo">
      <h3>Ejemplo: una clase grabada con OBS</h3>
      <p>Un recorrido típico de punta a punta. No hace falta pasar por todas las secciones.</p>
      <ol>
        <li><div><strong>Video / Audio.</strong> Recortás el video (por ejemplo, el principio y el final que sobran) y extraés el audio en MP3.</div></li>
        <li><div><strong>Transcripción.</strong> Con ese MP3 hacés la transcripción con Whisper, que separa a los hablantes y queda sincronizada sola. Si la clase la transcribiste con Tactiq, subís esa (ya viene con los nombres), la <strong>vinculás con el video</strong> y la sincronizás con una frase de referencia. Si tiene partes que quedaron fuera del video recortado, la recortás al subirla.</div></li>
        <li><div><strong>Hablantes.</strong> Etiquetás quién es quién, marcás cuál sos vos y corregís las líneas mal asignadas. Las voces quedan en el grupo para reconocerlas solas en las próximas clases.</div></li>
        <li><div><strong>Resultados.</strong> Bajás los subtítulos (SRT) para subirlos con el video a Drive. Si el video salió sin audio, usás la receta "Subtítulos de todo el video".</div></li>
        <li><div><strong>Resultados → Video</strong> <em>(opcional)</em>. Si vas a compartir la clase fuera del curso, usás "Publicar sin voces de alumnos": se silencian y lo que dijeron queda como subtítulo.</div></li>
      </ol>
    </div>

    <div class="guia-nota">
      <strong>Próximas etapas:</strong> clonar tu voz (XTTS / F5-TTS / RVC) con el dataset del paso 3 para pasar el guión convertido directo a audio con tu voz, y organizar los grupos de hablantes en niveles (categoría → área → subgrupo).
    </div>
  </div>
</div>

<main class="main">
<div class="main-inner">

<!-- ====================================================================== -->
<!-- PASO 1: CARGAR VIDEO O AUDIO · EXTRAER MP3 O RECORTAR                  -->
<!-- ====================================================================== -->
<section class="step-section activo" id="step-1">
  <div class="sec-head">
    <div>
      <div class="eyebrow">Paso 1</div>
      <h1>Video / Audio</h1>
      <p>Extraé el MP3 de un video o audio, recortá un tramo o dividilo en varias partes.</p>
    </div>
    <button type="button" class="btn-ghost btn-sm" onclick="irAPaso(2)">Ya tengo el audio: ir a Transcripción →</button>
  </div>
  <div class="uso"><span class="uso-tit">Usalo para</span><ul><li>Recortar un video</li><li>Recortar un audio</li><li>Extraer el audio (MP3) de un video</li><li>Dividir un video o audio en varias partes</li></ul></div>

  <div class="card">
    <div class="grid-2-60">
      <div>
        <div class="dropzone" id="zona-p1" style="min-height:260px">
          <div class="dz-icon"><svg><use href="#i-upload"/></svg></div>
          <div>
            <div class="dz-title" id="zona-p1-titulo">Arrastrá acá tu video o audio</div>
            <div class="dz-sub">o hacé clic para elegirlo · MP4, MKV, MOV, AVI, WEBM, WAV, M4A, MP3</div>
          </div>
          <input type="file" id="arch-p1" accept=".mkv,.mp4,.mov,.avi,.webm,.mp3,.wav,.m4a,.ogg,video/*,audio/*" multiple hidden>
        </div>

        <!-- Vista previa con minuto:segundo y marcas -->
        <div id="p1-preview" class="preview" style="display:none">
          <div class="row-between" style="margin-bottom:8px">
            <strong id="p1-prev-nombre" style="font-size:0.86rem;overflow-wrap:anywhere"></strong>
            <span class="badge" id="p1-prev-tipo"></span>
          </div>
          <video id="p1-prev-video" controls preload="metadata" playsinline></video>
          <audio id="p1-prev-audio" controls preload="metadata" style="display:none;width:100%"></audio>
          <div id="p1-prev-error" style="display:none;font-size:0.8rem;color:var(--warn);margin-top:6px">Este formato no se puede previsualizar en el navegador (igual se puede procesar). Podés escribir los tiempos a mano.</div>
          <div class="prev-reloj">
            <span class="t-grande" id="p1-prev-t">00:00</span>
            <span class="t-dur">/ <span id="p1-prev-dur">--:--</span></span>
            <span class="prev-saltos">
              <button type="button" class="btn-outline btn-xs" data-salto="-5">−5 s</button>
              <button type="button" class="btn-outline btn-xs" data-salto="-1">−1 s</button>
              <button type="button" class="btn-outline btn-xs" data-salto="1">+1 s</button>
              <button type="button" class="btn-outline btn-xs" data-salto="5">+5 s</button>
            </span>
          </div>
          <div class="timeline" id="p1-timeline" title="Hacé clic para ir a ese momento">
            <div class="tl-capa" id="p1-tl-marcas"></div>
            <div class="tl-cabezal" id="p1-tl-cabezal"></div>
          </div>
          <div class="row" style="gap:6px;margin-top:10px" id="p1-prev-acciones-tramo">
            <button type="button" class="btn-outline btn-sm" id="btn-p1-marcar-desde">⟦ Marcar Desde</button>
            <button type="button" class="btn-outline btn-sm" id="btn-p1-marcar-hasta">Marcar Hasta ⟧</button>
            <span style="font-size:0.76rem;color:var(--muted)">Pausá donde quieras cortar y tocá el botón.</span>
          </div>
          <div class="row" style="gap:6px;margin-top:10px;display:none" id="p1-prev-acciones-partes">
            <button type="button" class="btn-outline btn-sm" id="btn-p1-agregar-corte">✂ Agregar corte acá</button>
            <button type="button" class="btn-outline btn-sm" id="btn-p1-quitar-corte">Quitar el último</button>
            <span style="font-size:0.76rem;color:var(--muted)">Se suma a "En estos puntos".</span>
          </div>
        </div>
      </div>

      <div class="stack">
        <div class="campo">
          <label for="p1-modo-salida">¿Qué querés obtener?</label>
          <select id="p1-modo-salida">
            <option value="mp3" selected>Audio MP3</option>
            <option value="video_mp4">Video MP4 recortado</option>
            <option value="ambos">Ambos: MP3 + video MP4 recortado</option>
          </select>
        </div>

        <div class="campo">
          <label>Cómo cortar</label>
          <div class="segmentado" role="radiogroup" aria-label="Cómo cortar">
            <label><input type="radio" name="p1-tipo-corte" value="tramo" checked><span>Un tramo</span></label>
            <label><input type="radio" name="p1-tipo-corte" value="partes"><span>Dividir en partes</span></label>
          </div>
        </div>

        <div id="p1-split-box" class="panel" style="display:none">
          <div class="stack" style="gap:10px">
            <label class="check" style="align-items:center">
              <input type="radio" name="p1-split-modo" value="puntos" checked>
              <span style="flex-shrink:0;font-weight:600">En estos puntos</span>
            </label>
            <input type="text" id="p1-split-puntos" class="tiempo" placeholder="10:00, 25:30, 1:02:15" aria-label="Puntos de corte">
            <label class="check" style="align-items:center">
              <input type="radio" name="p1-split-modo" value="duracion">
              <span style="font-weight:600">Partes de duración fija, cada</span>
              <input type="text" id="p1-split-duracion" class="tiempo" placeholder="10:00" style="width:96px" aria-label="Duración de cada parte">
            </label>
            <label class="check" id="p1-split-exacto-box"><input type="checkbox" id="p1-split-exacto"><span>Cortes exactos al segundo en el video <span style="color:var(--muted)">(recodifica, más lento; si no, corta en el fotograma clave más cercano)</span></span></label>
            <div id="p1-split-preview" style="font-size:0.8rem;color:var(--primary-hover)"></div>
          </div>
        </div>

        <div>
          <label class="lbl" id="p1-lbl-tramo">Recorte <span style="font-weight:400;color:var(--muted)">(opcional · vacío = archivo completo)</span></label>
          <div class="grid-2" style="gap:10px">
            <div class="campo">
              <input id="p1-desde" class="tiempo" type="text" placeholder="Desde · 02:15" aria-label="Desde (minuto:segundo)">
            </div>
            <div class="campo">
              <input id="p1-hasta" class="tiempo" type="text" placeholder="Hasta · 45:20" aria-label="Hasta (minuto:segundo)">
            </div>
          </div>
        </div>

        <details class="plegable" style="margin:0">
          <summary>Opciones del MP3 <span class="hint">· 192 kbps, mono</span></summary>
          <div class="plegable-body stack" style="gap:10px">
            <div class="campo">
              <label for="p1-cal">Calidad</label>
              <select id="p1-cal">
                <option value="128k">128 kbps</option>
                <option value="192k" selected>192 kbps (recomendada)</option>
                <option value="320k">320 kbps (alta calidad)</option>
              </select>
            </div>
            <label class="check"><input type="checkbox" id="p1-mono" checked><span>Mono (recomendado para voz)</span></label>
            <div class="row">
              <label for="p1-mb" style="font-size:0.82rem;color:var(--text-2)">Tamaño máximo (MB):</label>
              <input id="p1-mb" type="number" min="1" step="0.5" placeholder="sin límite" style="width:110px">
            </div>
          </div>
        </details>

        <button type="button" class="btn-primary btn-block btn-lg" id="btn-p1-convertir" disabled>Procesar archivo(s)</button>
      </div>
    </div>

    <div id="p1-cola-box" style="display:none;margin-top:16px">
      <div class="row-between">
        <strong id="p1-cola-titulo" style="font-size:0.88rem">Archivos seleccionados (0)</strong>
        <button type="button" class="btn-danger-soft btn-xs" id="btn-p1-limpiar">Vaciar lista</button>
      </div>
      <ul class="cola-lista" id="p1-cola-lista"></ul>
    </div>

    <div id="p1-progreso"></div>
  </div>

  <div class="card">
    <div class="card-head" style="margin-bottom:10px">
      <div>
        <h3>Archivos ya guardados en <code>audios/</code></h3>
        <p>Recortá uno que ya tengas sin volver a subirlo (usa el Desde / Hasta de arriba).</p>
      </div>
    </div>
    <div class="row">
      <select id="p1-sel-existente" style="flex:1;min-width:220px;max-width:460px"></select>
      <button type="button" class="btn-outline btn-sm" id="btn-p1-ver-existente">Vista previa</button>
      <button type="button" class="btn-outline btn-sm" id="btn-p1-recortar-existente">Recortar este archivo</button>
    </div>
  </div>
</section>

<!-- ====================================================================== -->
<!-- PASO 2: TRANSCRIBIR CON IA O SUBIR TRANSCRIPCIÓN                       -->
<!-- ====================================================================== -->
<section class="step-section" id="step-2">
  <div class="sec-head">
    <div>
      <div class="eyebrow">Paso 2</div>
      <h1>Transcripción</h1>
      <p>Transcribí un audio o video con Whisper, subí una transcripción hecha en otro lado o abrí una que ya procesaste (al final).</p>
    </div>
  </div>
  <div class="uso"><span class="uso-tit">Usalo para</span><ul><li>Transcribir un video o un audio con Whisper, separando a los hablantes</li><li>Subir una transcripción hecha con otra herramienta (Tactiq, YouTube…)</li><li>Vincular la transcripción con su video y sincronizarla (por ej. si viene de Tactiq)</li><li>Recortar una transcripción (por ejemplo, las partes que quedaron fuera del video)</li></ul></div>

  <div class="grid-2">
    <!-- OPCIÓN A -->
    <div class="card" style="margin-bottom:0">
      <span class="card-tag">Opción A</span>
      <h2>Transcribir con Whisper</h2>
      <p class="sub-desc">Se transcribe en tu PC y separa las voces de los hablantes.</p>

      <div class="stack">
        <div class="dropzone dz-compacta" id="zona-p2-audio" role="button" tabindex="0">
          <div class="dz-icon"><svg><use href="#i-upload"/></svg></div>
          <div style="text-align:left">
            <div class="dz-title" id="lbl-p2-audio">Subí un audio o video</div>
            <div class="dz-sub">Arrastralo acá o hacé clic · MP3, WAV, M4A, MP4, MKV…</div>
          </div>
          <input type="file" id="arch-p2-audio" accept=".mp3,.wav,.m4a,.ogg,.flac,.aac,.mp4,.mkv,.mov,.webm,.avi,audio/*,video/*" hidden>
        </div>
        <div class="campo">
          <label for="p2-sel-mp3">o elegí uno de <code>audios/</code></label>
          <select id="p2-sel-mp3">
            <option value="">Cargando audios…</option>
          </select>
        </div>

        <div class="grid-2" style="gap:10px">
          <div class="campo">
            <label for="p2-modelo">Modelo</label>
            <select id="p2-modelo">
              <option value="base">Base (más rápido)</option>
              <option value="small" selected>Small (más preciso)</option>
            </select>
          </div>
          <div class="campo">
            <label for="p2-hablantes">Hablantes</label>
            <select id="p2-hablantes">
              <option value="0" selected>Automático</option>
              <option value="1">1 solo hablante</option>
              <option value="2">2 hablantes</option>
              <option value="3">3 hablantes</option>
              <option value="4">4 hablantes</option>
              <option value="5">5 hablantes</option>
              <option value="6">6 hablantes</option>
              <option value="7">7 hablantes</option>
              <option value="8">8 hablantes</option>
              <option value="9">9 hablantes</option>
              <option value="10">10 hablantes</option>
              <option value="11">11 hablantes</option>
              <option value="12">12 hablantes</option>
              <option value="13">13 hablantes</option>
              <option value="14">14 hablantes</option>
              <option value="15">15 hablantes</option>
              <option value="16">16 hablantes</option>
              <option value="17">17 hablantes</option>
              <option value="18">18 hablantes</option>
              <option value="19">19 hablantes</option>
              <option value="20">20 hablantes</option>
              <option value="21">21 hablantes</option>
              <option value="22">22 hablantes</option>
              <option value="23">23 hablantes</option>
              <option value="24">24 hablantes</option>
              <option value="25">25 hablantes</option>
              <option value="26">26 hablantes</option>
              <option value="27">27 hablantes</option>
              <option value="28">28 hablantes</option>
              <option value="29">29 hablantes</option>
              <option value="30">30 hablantes</option>
            </select>
          </div>
        </div>
      </div>

      <details class="plegable">
        <summary>Transcribir solo un tramo <span class="hint">· opcional</span></summary>
        <div class="plegable-body grid-2" style="gap:10px">
          <div class="campo">
            <label for="p2-desde">Desde</label>
            <input type="text" id="p2-desde" class="tiempo" placeholder="03:15 (vacío = inicio)">
          </div>
          <div class="campo">
            <label for="p2-hasta">Hasta</label>
            <input type="text" id="p2-hasta" class="tiempo" placeholder="28:40 (vacío = final)">
          </div>
        </div>
      </details>

      <button type="button" class="btn-primary btn-block btn-lg" id="btn-p2-transcribir">Transcribir y pasar al paso 3</button>
      <div id="p2-progreso-whisper"></div>
    </div>

    <!-- OPCIÓN B -->
    <div class="card" style="margin-bottom:0">
      <span class="card-tag">Opción B</span>
      <h2>Subir una transcripción ya hecha</h2>
      <p class="sub-desc">Detecta sola si es de Tactiq, YouTube, Whisper o SRT.</p>

      <div class="dropzone" id="zona-p2-trans" style="padding:20px">
        <div class="dz-icon"><svg><use href="#i-text"/></svg></div>
        <div class="dz-title" id="lbl-arch-trans">Arrastrá o elegí el archivo</div>
        <div class="dz-sub">TXT, SRT, VTT, DOCX o PDF</div>
        <input type="file" id="arch-p2-trans" accept=".txt,.srt,.vtt,.sbv,.docx,.pdf,.json" hidden>
      </div>

      <div class="campo" style="margin-top:14px">
        <label for="p2-trans-mp3">Vincular con su audio o video <span style="font-weight:400;color:var(--muted)">(opcional, para escuchar las voces)</span></label>
        <select id="p2-trans-mp3">
          <option value="">Sin audio (solo transcripción y subtítulos)</option>
        </select>
      </div>

      <details class="plegable">
        <summary>Recortar la transcripción al subirla <span class="hint">· opcional</span></summary>
        <div class="plegable-body">
          <div class="grid-2" style="gap:10px;margin-bottom:10px">
            <div class="campo">
              <label for="p2-trans-desde">Desde</label>
              <input type="text" id="p2-trans-desde" class="tiempo" placeholder="05:00">
            </div>
            <div class="campo">
              <label for="p2-trans-hasta">Hasta</label>
              <input type="text" id="p2-trans-hasta" class="tiempo" placeholder="40:00">
            </div>
          </div>
          <label class="check"><input type="checkbox" id="p2-trans-reset"><span>Reiniciar los tiempos a <code>00:00</code> desde el inicio del recorte</span></label>
        </div>
      </details>

      <button type="button" class="btn-secondary btn-block btn-lg" id="btn-p2-subir-trans" disabled>Cargar y vincular con el video</button>
      <div id="p2-res-import"></div>
    </div>
  </div>
  <!-- Vincular y sincronizar la transcripción abierta -->
  <div class="card destacada" id="p2-vinculo" style="display:none;margin-top:16px">
    <div class="paso-cab">
      <span class="paso-n"><svg width="16" height="16"><use href="#i-link"/></svg></span>
      <div style="flex:1">
        <h2>Vincular con su video y sincronizar</h2>
        <p>Así lo que dice la transcripción coincide con el video. Queda guardado en la transcripción.</p>
      </div>
    </div>
    <div class="row" style="margin-bottom:14px">
      <label for="p4-sel-sesion-activa" style="font-size:0.8rem;font-weight:600">Transcripción abierta</label>
      <select id="p4-sel-sesion-activa" style="flex:1;min-width:240px;max-width:520px"></select>
      <span id="p2v-estado" class="badge"></span>
    </div>
    <div class="grid-2" style="align-items:start">
      <div>
        <div class="pub-video-wrap">
          <video id="p4-video" controls preload="metadata" playsinline></video>
          <div class="sub-overlay pub-overlay"><span id="p4-sub-live" style="display:none"></span></div>
        </div>
        <div class="prev-reloj" style="margin-bottom:0">
          <span class="t-grande" id="p2v-t">00:00</span>
          <span class="t-dur">/ <span id="p2v-dur">--:--</span></span>
          <span class="prev-saltos">
            <button type="button" class="btn-outline btn-xs" data-salto-v="-5">−5 s</button>
            <button type="button" class="btn-outline btn-xs" data-salto-v="-1">−1 s</button>
            <button type="button" class="btn-outline btn-xs" data-salto-v="1">+1 s</button>
            <button type="button" class="btn-outline btn-xs" data-salto-v="5">+5 s</button>
          </span>
        </div>
        <div id="p2v-sin-media" class="vacio-mini">Elegí a la derecha el video o audio de esta clase para verlo acá.</div>
      </div>

      <div class="stack" style="gap:14px">
        <div class="campo">
          <label for="p2v-media">1 · Video o audio de esta clase</label>
          <div class="row" style="flex-wrap:nowrap">
            <select id="p2v-media" style="flex:1"></select>
            <input type="file" id="p2v-arch" accept=".mp4,.mkv,.mov,.webm,.avi,.mp3,.wav,.m4a,.ogg,video/*,audio/*" hidden>
            <button type="button" class="btn-outline btn-sm" onclick="document.getElementById('p2v-arch').click()">Subir</button>
          </div>
          <div id="p2v-media-info" style="font-size:0.78rem;color:var(--muted);margin-top:4px"></div>
        </div>

        <div id="p2v-sync" class="panel">
          <span class="panel-titulo">2 · Sincronizar</span>
          <p class="panel-desc" id="p2v-sync-ayuda">Elegí una frase, buscá en el video el momento en que se dice, pausá y tocá <strong>Hacer coincidir</strong>. Se ajusta toda la transcripción, hacia atrás y hacia adelante.</p>
          <div class="campo" style="margin-bottom:8px">
            <label for="p4-sel-frase-ref">Frase de referencia</label>
            <select id="p4-sel-frase-ref"></select>
          </div>
          <div class="row" style="gap:6px;margin-bottom:10px">
            <button type="button" class="btn-outline btn-xs" id="btn-p2v-ir-frase" title="Lleva el video al momento donde la transcripción cree que está la frase">Ir a la frase</button>
            <input type="text" id="p4-t-video-ancla" class="tiempo" placeholder="se dice en 02:35" style="width:130px" aria-label="Minuto del video en que se dice la frase">
            <button type="button" class="btn-outline btn-xs" id="btn-p4-tomar-ancla-video" title="Usar el momento donde está pausado el video">Tomar del video</button>
            <button type="button" class="btn-primary btn-sm" id="btn-p4-sync-video">Hacer coincidir</button>
          </div>
          <div class="row" style="gap:6px">
            <label for="p4-offset-seg" style="font-size:0.8rem;font-weight:600">Desfase (s)</label>
            <input type="number" id="p4-offset-seg" step="0.5" value="0" class="tiempo" style="width:90px">
            <button type="button" class="btn-outline btn-xs" onclick="ajustarOffsetSub(-1)">−1 s</button>
            <button type="button" class="btn-outline btn-xs" onclick="ajustarOffsetSub(1)">+1 s</button>
            <button type="button" class="btn-outline btn-xs" onclick="invertirSignoOffsetSub()" title="Cambia el signo">±</button>
            <button type="button" class="btn-outline btn-xs" onclick="ajustarOffsetSub(0, true)">0</button>
          </div>
          <div id="p2v-explica" style="font-size:0.78rem;color:var(--text-2);margin-top:8px"></div>
          <button type="button" class="btn-secondary btn-sm btn-block" id="btn-p2v-ok" style="margin-top:10px">✓ Está sincronizada</button>
        </div>

        <div class="row-between">
          <span style="font-size:0.78rem;color:var(--muted)" id="p2v-guardado"></span>
          <button type="button" class="btn-primary" onclick="irAPaso(3)">Seguir: Hablantes →</button>
        </div>
      </div>
    </div>
  </div>

  <div class="card" id="card-sesiones-previas" style="margin-top:16px">
    <div class="card-head" style="margin-bottom:10px">
      <div>
        <h2>Transcripciones guardadas</h2>
        <p>Abrí una para vincularla con su video, revisar los hablantes o ir directo a Resultados.</p>
      </div>
    </div>
    <div id="lista-sesiones-guardadas" style="display:flex;flex-direction:column;gap:8px;max-height:300px;overflow-y:auto">
      <span style="font-size:0.85rem;color:var(--muted)">Cargando transcripciones guardadas…</span>
    </div>
  </div>

</section>

<!-- ====================================================================== -->
<!-- PASO 3: HABLANTES POR GRUPO, AUTO-MATCHEO Y MI VOZ                     -->
<!-- ====================================================================== -->
<section class="step-section" id="step-3">
  <div class="sec-head">
    <div>
      <div class="eyebrow">Paso 3</div>
      <h1>Hablantes y mi voz</h1>
      <p>Identificá quién es quién y revisá la transcripción. Después, en Resultados, te llevás lo que necesites.</p>
    </div>
  </div>
  <div class="uso"><span class="uso-tit">Usalo para</span><ul><li>Poner nombre a cada voz (a mano, con Tactiq, por presentaciones o con una lista)</li><li>Marcar cuál es tu voz</li><li>Guardar las voces para reconocerlas solas en próximas clases</li><li>Corregir líneas con la persona equivocada o el texto mal transcripto</li></ul></div>

  <div id="p3-vacio" class="card vacio">
    <div class="vacio-icon"><svg><use href="#i-users"/></svg></div>
    <h3>No hay ninguna transcripción abierta</h3>
    <p>Abrí o creá una en el paso 2 para etiquetar hablantes, matchear voces o extraer tu voz.</p>
    <button type="button" class="btn-primary" onclick="irAPaso(2)">Ir a Transcripción</button>
  </div>

  <div id="p3-contenido" style="display:none">
    <!-- Barra de contexto: qué transcripción está abierta -->
    <div class="card ctx-bar">
      <div class="ctx-main">
        <div class="ctx-tit"><span class="ctx-label">Transcripción abierta</span><strong id="p3-ctx-nombre">—</strong></div>
        <div class="ctx-datos" id="p3-ctx-datos"></div>
      </div>
      <div class="row" style="gap:6px">
        <label for="p3-sel-grupo" style="font-size:0.78rem;color:var(--muted)">Grupo</label>
        <select id="p3-sel-grupo" style="width:auto;min-width:160px;padding:6px 8px;font-size:0.84rem"></select>
        <button type="button" class="btn-outline btn-xs" id="btn-nuevo-grupo" title="Crear un grupo nuevo (por ejemplo, otro curso)">+ Grupo</button>
        <button type="button" class="btn-ghost btn-xs" onclick="irAPaso(2)">Cambiar transcripción</button>
      </div>
      <span class="badge badge-azul" id="badge-info-grupo" style="display:none"></span>
    </div>

    <!-- Diagnóstico (compatibilidad: el texto se resume en la barra de contexto) -->
    <div style="display:none">
      <div id="diag-hablantes-box"><p id="diag-hablantes-txt"></p></div>
      <div id="diag-mivoz-box"><p id="diag-mivoz-txt"></p></div>
    </div>

    <!-- Reproductor -->
    <div class="player-sticky" id="p3-player-bar">
      <audio id="p3-audio" controls preload="metadata"></audio>
      <div style="font-size:0.82rem;color:#d1d5db" id="p3-player-info">Tocá ▶ en una voz o en una línea para escucharla.</div>
    </div>

    <!-- ============ 1. ¿QUIÉN ES QUIÉN? ============ -->
    <div class="card paso-bloque">
      <div class="paso-cab">
        <span class="paso-n">1</span>
        <div>
          <h2>¿Quién es quién?</h2>
          <p>Escuchá cada voz, escribí su nombre y marcá cuál sos vos. Los nombres se guardan en el grupo para reconocerlos solos la próxima vez.</p>
        </div>
      </div>

      <div id="p3-aviso-media" class="aviso-mini" style="margin-bottom:12px"></div>
      <details class="plegable" id="p3-auto-box">
        <summary>✨ Completar los nombres automáticamente <span class="hint">· con Tactiq, por presentaciones o con una lista</span></summary>
        <div class="plegable-body">
          <div class="grid-3">
            <div class="panel">
              <span class="panel-titulo">Con la transcripción de Tactiq</span>
              <p class="panel-desc">La recomendada: toma los nombres de Tactiq y los asigna a cada voz, aunque los tiempos estén corridos.</p>
              <input type="file" id="arch-tactiq-sync" accept=".txt,.docx,.srt,.vtt,.json" hidden>
              <button type="button" class="btn-primary btn-sm btn-block" onclick="document.getElementById('arch-tactiq-sync').click()">Subir Tactiq</button>
              <div id="res-tactiq-sync" style="font-size:0.79rem;margin-top:6px"></div>
            </div>
            <div class="panel">
              <span class="panel-titulo">Por presentaciones</span>
              <p class="panel-desc">Busca cuando alguien dice <em>"soy…"</em> o <em>"mi nombre es…"</em>.</p>
              <button type="button" class="btn-outline btn-sm btn-block" id="btn-auto-presentacion">Buscar presentaciones</button>
              <div id="res-auto-presentacion" style="font-size:0.79rem;margin-top:6px"></div>
            </div>
            <div class="panel">
              <span class="panel-titulo">Lista de nombres del grupo</span>
              <p class="panel-desc">Cargá los nombres del curso (Excel, CSV o TXT) y te aparecen para elegir al escribir.</p>
              <div class="row" style="gap:6px">
                <input type="file" id="arch-listado-nombres" accept=".xlsx,.csv,.txt" hidden>
                <button type="button" class="btn-outline btn-sm" onclick="document.getElementById('arch-listado-nombres').click()">Subir lista</button>
                <button type="button" class="btn-outline btn-sm" id="btn-pegar-nombres">Pegar nombres</button>
              </div>
              <div id="chips-nombres-grupo" class="chips" style="margin-top:8px;max-height:90px;overflow-y:auto"></div>
            </div>
          </div>
        </div>
      </details>

      <datalist id="p3-dl-nombres"></datalist>
      <div id="p3-hablantes-grid" class="stack" style="gap:6px"></div>

      <div class="row-between" style="margin-top:12px">
        <div class="row" style="gap:6px;font-size:0.82rem;color:var(--text-2)">
          <span>¿Voces mezcladas o repetidas? Separar en</span>
          <button type="button" class="btn-outline btn-xs btn-recluster-k" data-k="2">2</button>
          <button type="button" class="btn-outline btn-xs btn-recluster-k" data-k="3">3</button>
          <button type="button" class="btn-outline btn-xs btn-recluster-k" data-k="4">4</button>
          <button type="button" class="btn-outline btn-xs btn-recluster-k" data-k="5">5</button>
          <select id="p3-k-otro" aria-label="Otra cantidad de voces" style="width:auto;padding:3px 6px;font-size:0.76rem">
            <option value="6">6</option>
            <option value="7">7</option>
            <option value="8">8</option>
            <option value="9">9</option>
            <option value="10">10</option>
            <option value="11">11</option>
            <option value="12">12</option>
            <option value="13">13</option>
            <option value="14">14</option>
            <option value="15">15</option>
            <option value="16">16</option>
            <option value="17">17</option>
            <option value="18">18</option>
            <option value="19">19</option>
            <option value="20">20</option>
            <option value="21">21</option>
            <option value="22">22</option>
            <option value="23">23</option>
            <option value="24">24</option>
            <option value="25">25</option>
            <option value="26">26</option>
            <option value="27">27</option>
            <option value="28">28</option>
            <option value="29">29</option>
            <option value="30">30</option>
          </select>
          <button type="button" class="btn-outline btn-xs btn-recluster-k" id="btn-p3-k-otro" data-k="6">voces</button>
        </div>
        <button type="button" class="btn-ghost btn-xs" id="btn-p3-nuevo-spk">+ Agregar una voz</button>
      </div>
      <!-- Compatibilidad con funciones anteriores -->
      <span id="btn-p3-silenciar-todos" hidden></span><span id="btn-p3-activar-todos" hidden></span>
    </div>

    <!-- ============ 2. REVISAR LA TRANSCRIPCIÓN ============ -->
    <div class="card paso-bloque">
      <div class="paso-cab">
        <span class="paso-n">2</span>
        <div>
          <h2>Revisar la transcripción <span style="font-weight:400;color:var(--muted);font-size:0.86rem">· <span id="p3-cont-seg">0</span> líneas</span></h2>
          <p>Si una línea quedó con la persona equivocada, cambiala en su nombre; el texto se corrige haciendo clic. Todo se guarda solo. Con ⛳ y 🏁 marcás un tramo para Resultados.</p>
        </div>
      </div>
      <div class="row" style="margin-bottom:10px">
        <input type="text" id="p3-buscar" placeholder="Buscar palabra o frase…" style="flex:1;min-width:180px;max-width:320px">
        <select id="p3-filtro-spk" style="width:200px"><option value="">Todas las voces</option></select>
        <label class="check" style="font-size:0.8rem"><input type="checkbox" id="p3-chk-ocultar-fuera"><span>Solo las líneas del tramo</span></label>
      </div>
      <div class="segmentos-wrap" id="p3-seg-wrap"></div>
      <div class="row-between" style="margin-top:12px">
        <span style="font-size:0.78rem;color:var(--muted)">Las correcciones se guardan solas.</span>
        <button type="button" class="btn-primary" onclick="irAPaso(4)">Seguir: Resultados →</button>
      </div>
    </div>

  </div>
</section>

<!-- ====================================================================== -->
<!-- PASO 4: SUBTÍTULOS Y SINCRONIZACIÓN CON VIDEO                          -->
<!-- ====================================================================== -->
<section class="step-section" id="step-4">
  <div class="sec-head">
    <div>
      <div class="eyebrow">Paso 4</div>
      <h1>Resultados</h1>
      <p>Con la transcripción vinculada, sincronizada y con los hablantes identificados, llevate lo que necesites.</p>
    </div>
  </div>
  <div class="uso"><span class="uso-tit">Usalo para</span><ul><li>Tu voz limpia para entrenar o clonar</li><li>Audio sin ciertas voces</li><li>Transcripción o subtítulos (SRT / VTT) del tramo que quieras</li><li>Video con voces silenciadas y/o subtítulos (por ej. para publicar sin los alumnos)</li><li>Recortar el video o unirle subtítulos</li></ul></div>

  <div id="p4-vacio" class="card vacio">
    <div class="vacio-icon"><svg><use href="#i-cc"/></svg></div>
    <h3>Primero abrí una transcripción</h3>
    <p>Creala o abrila en el paso 2 (Transcripción), vinculala con su video o audio y, si hace falta, sincronizala.</p>
    <button type="button" class="btn-primary" onclick="irAPaso(2)">Ir a Transcripción</button>
  </div>

  <div id="p4-contenido" style="display:none">
    <!-- Material + reproductor + tramo -->
    <div class="card">
      <div class="res-top">
        <div class="stack" style="gap:12px">
          <div>
            <span class="ctx-label">Material</span>
            <div id="res-material" class="res-material"></div>
          </div>
          <div>
            <span class="ctx-label">Tramo</span>
            <div style="margin-top:6px">
              <div class="panel">
        <div class="row" style="gap:10px;align-items:flex-end">
          <div class="campo">
            <label for="p3-t-ini">Desde</label>
            <div class="input-con-btn">
              <input type="text" id="p3-t-ini" class="tiempo" placeholder="00:00" style="width:110px">
              <button type="button" class="btn-outline btn-xs" id="btn-p3-tomar-ini-audio" title="Usar el segundo actual del reproductor">⟦ Actual</button>
            </div>
          </div>
          <div class="campo">
            <label for="p3-t-fin">Hasta</label>
            <div class="input-con-btn">
              <input type="text" id="p3-t-fin" class="tiempo" placeholder="final" style="width:110px">
              <button type="button" class="btn-outline btn-xs" id="btn-p3-tomar-fin-audio" title="Usar el segundo actual del reproductor">Actual ⟧</button>
            </div>
          </div>
          <button type="button" class="btn-primary btn-sm" id="btn-p3-aplicar-rango">Aplicar</button>
          <button type="button" class="btn-outline btn-sm" id="btn-p3-limpiar-rango">Todo el archivo</button>
          <span id="p3-badge-rango" class="badge badge-azul">Tramo: todo el archivo</span>
        </div>
        <div class="timeline" id="p3-timeline" style="margin-top:10px;height:22px" title="Hacé clic para ir a ese momento">
          <div class="tl-capa" id="p3-tl-marcas"></div>
          <div class="tl-cabezal" id="p3-tl-cabezal"></div>
        </div>
        <details class="plegable" style="margin:10px 0 0;background:#fff">
          <summary>Buscar el tramo por frase <span class="hint">· de inicio y de fin</span></summary>
          <div class="plegable-body">
            <div class="grid-2" style="gap:8px;margin-bottom:8px">
              <div class="campo"><input type="text" id="p3-f-ini" placeholder='Frase de inicio, ej. "empezamos"'><div id="p3-pill-ini"></div></div>
              <div class="campo"><input type="text" id="p3-f-fin" placeholder='Frase de fin, ej. "hasta acá"'><div id="p3-pill-fin"></div></div>
            </div>
            <button type="button" class="btn-outline btn-xs" id="btn-p3-buscar-frases">Buscar y completar tiempos</button>
          </div>
        </details>
      </div>

            </div>
          </div>
        </div>
        <div id="pub-preview" class="preview" style="margin-top:0">
          <div class="pub-video-wrap">
            <video id="pub-audio" controls preload="metadata" playsinline></video>
            <div class="sub-overlay pub-overlay"><span id="pub-sub-live" style="display:none"></span></div>
            <span class="pub-mute-tag" id="pub-mute-tag" style="display:none">🔇 silenciada</span>
          </div>
          <div class="prev-reloj">
            <span class="t-grande" id="pub-prev-t">00:00</span>
            <span class="t-dur">/ <span id="pub-prev-dur">--:--</span></span>
            <span class="prev-saltos">
              <button type="button" class="btn-outline btn-xs" data-salto-pub="-5">−5 s</button>
              <button type="button" class="btn-outline btn-xs" data-salto-pub="5">+5 s</button>
            </span>
          </div>
          <div class="timeline" id="pub-timeline" title="Hacé clic para ir a ese momento">
            <div class="tl-capa" id="pub-tl-marcas"></div>
            <div class="tl-cabezal" id="pub-tl-cabezal"></div>
          </div>
          <div class="tl-leyenda" id="pub-tl-leyenda" style="display:none">
            <span><i style="background:rgba(220,38,38,0.55)"></i>silenciada + subtítulo</span>
            <span><i style="background:rgba(37,99,235,0.55)"></i>solo subtítulo</span>
          </div>
          <div id="pub-audio-info" class="pub-ahora">Dale play para revisar.</div>
          <label class="check" id="pub-simular-box" style="font-size:0.8rem;margin-top:6px"><input type="checkbox" id="pub-simular" checked><span>Simular lo elegido en "Video" (silencio y subtítulos, aproximado)</span></label>
        </div>
      </div>
    </div>

    <!-- AUDIO + TEXTO -->
    <div class="res-grupo-tit">Audio y texto</div>
    <div class="card">
      <div class="salidas-grid">
        <!-- A. Mi voz limpia -->
        <div class="salida">
          <div class="salida-tit">🎙️ Mi voz limpia</div>
          <p class="salida-desc">Solo tus fragmentos, unidos, para entrenar o clonar tu voz.</p>
          <div id="p3-mivoz-aviso" class="aviso-mini"></div>
          <div class="campo"><label for="p3-fmt-audio">Formato</label>
            <select id="p3-fmt-audio">
              <option value="mp3" selected>MP3</option>
              <option value="wav">WAV 44.1 kHz</option>
              <option value="zip">Dataset ZIP (clips + metadata.csv)</option>
            </select>
          </div>
          <div class="grid-2" style="gap:8px">
            <div class="campo"><label for="p3-max-min">Duración</label>
              <select id="p3-max-min">
                <option value="0" selected>Todo</option><option value="2">Mejores 2 min</option>
                <option value="5">Mejores 5 min</option><option value="10">Mejores 10 min</option>
              </select>
            </div>
            <div class="campo"><label for="p3-min-seg">Frases cortas</label>
              <select id="p3-min-seg">
                <option value="0">Todas</option><option value="1.0" selected>Sin &lt; 1 s</option><option value="2.0">Sin &lt; 2 s</option>
              </select>
            </div>
          </div>
          <div id="p3-resumen-audio" class="salida-resumen"></div>
          <button type="button" class="btn-primary btn-block" id="btn-p3-export-audio">Generar mi voz</button>
          <div id="p3-res-audio"></div>
        </div>

        <!-- B. Audio sin ciertas voces -->
        <div class="salida">
          <div class="salida-tit">🔇 Audio sin ciertas voces</div>
          <p class="salida-desc">Sacá o silenciá a quienes elijas. Sin nadie marcado, recorta el audio al tramo.</p>
          <div class="row" style="gap:4px;margin-bottom:6px">
            <button type="button" class="btn-outline btn-xs" id="btn-p3b-todos-menos-yo" title="Marca a todos menos a vos: queda solo tu voz">Todos menos yo</button>
            <button type="button" class="btn-outline btn-xs" id="btn-p3b-solo-yo" title="Marca solo tu voz: se saca o silencia tu voz y quedan los demás">Solo yo</button>
            <button type="button" class="btn-outline btn-xs" id="btn-p3b-ninguno">Nadie</button>
          </div>
          <div id="p3b-voces" class="voces-check"></div>
          <div class="grid-2" style="gap:8px">
            <div class="campo"><label for="p3b-modo">Cómo</label>
              <select id="p3b-modo">
                <option value="silenciar" selected>Silenciar</option>
                <option value="recortar">Quitar (acorta)</option>
              </select>
            </div>
            <div class="campo"><label for="p3b-fmt">Formato</label>
              <select id="p3b-fmt"><option value="mp3" selected>MP3</option><option value="wav">WAV</option></select>
            </div>
          </div>
          <div id="p3b-resumen" class="salida-resumen"></div>
          <button type="button" class="btn-primary btn-block" id="btn-p3b-export">Generar audio</button>
          <div id="p3b-res"></div>
        </div>

        <!-- C. Transcripción -->
        <div class="salida">
          <div class="salida-tit">📄 Transcripción</div>
          <p class="salida-desc">Del tramo elegido, con los nombres de cada voz.</p>
          <label class="check" style="margin-bottom:8px"><input type="checkbox" id="p3-chk-reset-reloj" checked><span>Que el tramo empiece en <code>00:00</code> (si también recortaste el audio o el video)</span></label>
          <div class="stack" style="gap:6px">
            <button type="button" class="btn-primary btn-sm btn-block" onclick="descargarTranscripcionRecortadaPaso3('txt_ts')">TXT con tiempos</button>
            <button type="button" class="btn-outline btn-sm btn-block" onclick="descargarTranscripcionRecortadaPaso3('txt_limpio')">TXT solo texto</button>
            <button type="button" class="btn-outline btn-sm btn-block" onclick="descargarTranscripcionRecortadaPaso3('srt')">Subtítulos SRT</button>
          </div>
          <button type="button" class="btn-danger-soft btn-xs" id="btn-p3-recortar-sesion-perm" style="margin-top:10px" title="Borra de esta transcripción lo que está fuera del tramo">Dejar solo el tramo en esta transcripción</button>
        </div>

      </div>
    </div>

    <!-- SUBTÍTULOS -->
    <div class="card">
      <div class="card-head">
        <div>
          <h2>📝 Subtítulos para el video</h2>
          <p>Ya sincronizados con el video vinculado. Para subir a Drive, YouTube o VLC.</p>
        </div>
        <div class="row" style="gap:6px">
          <button type="button" class="btn-primary btn-sm" onclick="descargarSubtitulosPaso4('srt')">Descargar SRT</button>
          <button type="button" class="btn-outline btn-sm" onclick="descargarSubtitulosPaso4('vtt')">VTT</button>
          <button type="button" class="btn-outline btn-sm" onclick="descargarSubtitulosPaso4('txt_ts')">TXT con tiempos</button>
        </div>
      </div>
      <div class="grid-2" style="align-items:start">
        <div class="stack" style="gap:8px">
          <label class="check"><input type="checkbox" id="p4-inc-nom" checked><span>Con el nombre de quien habla (<code>[Nombre]: texto</code>)</span></label>
          <label class="check"><input type="checkbox" id="p4-chk-bloque-00"><span>Si el video empieza antes que la transcripción, agregar un cartel inicial desde <code>00:00</code></span></label>
          <label class="check" style="display:none"><input type="checkbox" id="p4-inc-sil"><span>Incluir líneas excluidas</span></label>
        </div>
        <div class="grid-2" style="gap:10px">
          <div class="campo">
            <label for="p4-max-chars">Caracteres por cartel</label>
            <select id="p4-max-chars">
              <option value="70">70 (2 líneas de 35)</option>
              <option value="84" selected>84 (estándar)</option>
              <option value="100">100 (2 líneas de 50)</option>
            </select>
          </div>
          <div class="campo">
            <label for="p4-max-dur">Duración en pantalla</label>
            <select id="p4-max-dur">
              <option value="4.5">1,2 a 4,5 s</option>
              <option value="6.0" selected>1,2 a 6 s (estándar)</option>
              <option value="8.0">1,2 a 8 s</option>
            </select>
          </div>
        </div>
      </div>
      <p style="font-size:0.78rem;color:var(--muted);margin:10px 0 0">Si se elige un tramo arriba, los subtítulos son solo de ese tramo (con los tiempos del video completo).</p>
    </div>

    <!-- VIDEO -->
    <div class="res-grupo-tit">Video</div>
    <div class="card" id="res-video-card">
      <div id="res-video-aviso" class="aviso-mini" style="margin-bottom:12px"></div>
      <div class="card-head" style="margin-bottom:10px">
        <div>
          <h2>🎬 Silenciar o subtitular voces</h2>
          <p>Usa la transcripción sincronizada con su video. Elegí una receta o decidí voz por voz y mirá el efecto en la vista previa de arriba. <span id="pub-resumen-video"></span></p>
        </div>
      </div>
      <div class="recetas" role="group" aria-label="Recetas">
        <span class="recetas-tit">Recetas</span>
        <button type="button" class="receta" data-receta-pub="publicar"><strong>Publicar sin voces de alumnos</strong><span>Los silencia y los subtitula como "Alumno 1, 2…"</span></button>
        <button type="button" class="receta" data-receta-pub="sub-alumnos"><strong>Subtítulos de los alumnos</strong><span>No se los escuchó: se agregan sus frases con nombre</span></button>
        <button type="button" class="receta" data-receta-pub="sub-todos"><strong>Subtítulos de todo el video</strong><span>El video salió sin audio: todos con nombre</span></button>
        <button type="button" class="receta" data-receta-pub="igualar"><strong>Solo igualar el volumen</strong><span>Tu voz y la de los alumnos parejas, sin subtítulos</span></button>
      </div>
      <div class="row" style="gap:6px;margin-bottom:10px">
        <span style="font-size:0.75rem;color:var(--muted)">A los demás (todos menos vos):</span>
        <button type="button" class="btn-outline btn-xs" data-bulk="demas:silenciar">Silenciar</button>
        <button type="button" class="btn-outline btn-xs" data-bulk="demas:subtitulo">Solo subtítulo</button>
        <span style="width:1px;height:18px;background:var(--border)"></span>
        <button type="button" class="btn-outline btn-xs" data-bulk="todos:silenciar">Silenciar a todos</button>
        <button type="button" class="btn-outline btn-xs" data-bulk="todos:normal">Reiniciar</button>
      </div>
      <div id="pub-lista-voces" class="voces-grid"></div>

      <hr class="divider">
      <div class="grid-2" style="align-items:start">
        <div class="stack" style="gap:12px">
          <div class="campo">
            <label>Nombre en los subtítulos</label>
            <div class="segmentado" style="grid-template-columns:1fr 1fr 1fr">
              <label><input type="radio" name="pub-nombres" value="anon" checked><span>Anonimizado</span></label>
              <label><input type="radio" name="pub-nombres" value="real"><span>Nombre real</span></label>
              <label><input type="radio" name="pub-nombres" value="ninguno"><span>Sin nombre</span></label>
            </div>
          </div>
          <div class="campo" id="pub-box-prefijo">
            <label for="pub-prefijo">Etiqueta anónima</label>
            <input type="text" id="pub-prefijo" value="Alumno" placeholder="Alumno" style="max-width:220px">
            <div style="font-size:0.78rem;color:var(--muted);margin-top:4px">Queda "Alumno 1", "Alumno 2"… en el orden en que hablan.</div>
          </div>
        </div>
        <div class="stack" style="gap:10px">
          <label class="check"><input type="checkbox" id="pub-igualar" checked><span><strong>Igualar el volumen entre voces</strong> (sube a quien habla bajo y baja a quien habla alto)</span></label>
          <label class="check"><input type="checkbox" id="pub-nivelar" checked><span><strong>Nivelar el volumen general</strong> (volumen estándar en todo el video)</span></label>
          <label class="check"><input type="checkbox" id="pub-margen" checked><span>Margen de seguridad: silenciar también 0,2 s antes y después</span></label>
        </div>
      </div>
      <div class="row-between" style="margin-top:14px">
        <span id="pub-resumen-sel" style="font-size:0.84rem;color:var(--text-2)"></span>
        <button type="button" class="btn-primary btn-lg" id="btn-pub-generar">Generar video</button>
      </div>
      <div id="pub-progreso-render"></div>
    </div>

    <div class="card">
      <h3>✂️ Recortar el video y/o unirle subtítulos</h3>
      <p class="sub-desc">Usa el tramo de arriba (vacío = video entero) y los subtítulos ya sincronizados de esta transcripción.</p>
      <div class="row">
        <input type="hidden" id="p4-t-ini"><input type="hidden" id="p4-t-fin">
        <select id="p4-modo-unir" style="flex:1;min-width:220px;max-width:420px">
          <option value="soft" selected>Unir subtítulos como pista (rápido)</option>
          <option value="burn">Subtítulos quemados en la imagen</option>
          <option value="solo_recortar_video">Solo recortar el video</option>
        </select>
        <button type="button" class="btn-secondary btn-sm" id="btn-p4-unir-video">Procesar MP4</button>
      </div>
      <div id="p4-res-unir" style="margin-top:8px;font-size:0.84rem"></div>
    </div>
  </div>
</section>


<!-- ====================================================================== -->
<!-- PESTAÑA 5: MI ESTILO (ML) Y GUIONES                                     -->
<!-- ====================================================================== -->
<section class="step-section" id="step-5">
  <div class="sec-head">
    <div>
      <div class="eyebrow">Herramientas IA</div>
      <h1>Mi estilo y guiones</h1>
      <p>Entrená un modelo con tus transcripciones y reescribí cualquier guión con tu forma de hablar.</p>
    </div>
    <span class="badge badge-azul" id="p5-badge-estado" style="font-size:0.8rem;padding:6px 12px">Cargando estado del modelo…</span>
  </div>
  <div class="uso"><span class="uso-tit">Usalo para</span><ul><li>Entrenar un modelo con tu forma de hablar a partir de tus clases</li><li>Reescribir un guion o un texto formal como lo dirías vos</li></ul></div>

  <div class="grid-2" style="align-items:start">
    <!-- Convertir guión (lo más usado, primero) -->
    <div class="card" style="margin:0">
      <span class="card-tag">Usar</span>
      <h2>Convertir un guión a mi estilo</h2>
      <p class="sub-desc">Pegá un guión formal, notas o cualquier texto neutro.</p>

      <div class="campo" style="margin-bottom:12px">
        <div class="row-between" style="margin-bottom:5px">
          <label for="p5-txt-guion" style="margin:0">Texto de entrada</label>
          <button type="button" class="btn-ghost btn-xs" id="btn-p5-ejemplo-guion">Cargar ejemplo</button>
        </div>
        <textarea id="p5-txt-guion" rows="8" placeholder="Ej.: 'A continuación veremos cómo definir una función en Python. Es importante destacar que puedes pasarle múltiples parámetros…'"></textarea>
      </div>

      <div class="grid-2" style="gap:10px;margin-bottom:12px">
        <div class="campo">
          <label for="p5-intensidad">Intensidad</label>
          <select id="p5-intensidad">
            <option value="sutil">Sutil (voseo y vocabulario, pocas muletillas)</option>
            <option value="equilibrado" selected>Equilibrado (recomendado)</option>
            <option value="conversacional">Muy conversacional (como en vivo)</option>
          </select>
        </div>
        <div class="campo">
          <label for="p5-formato-out">Formato de salida</label>
          <select id="p5-formato-out">
            <option value="fluido" selected>Párrafos fluidos</option>
            <option value="teleprompter">Frases cortas (teleprompter / XTTS)</option>
          </select>
        </div>
      </div>

      <label class="check" style="margin-bottom:14px"><input type="checkbox" id="p5-chk-ollama"><span>Usar también Ollama local (Llama3 / Mistral en <code>localhost:11434</code>)</span></label>

      <button type="button" class="btn-primary btn-block btn-lg" id="btn-p5-convertir">Convertir a mi estilo</button>

      <div id="p5-box-resultado" style="display:none;margin-top:18px">
        <div class="row-between" style="margin-bottom:6px">
          <strong style="color:var(--ok);font-size:0.92rem">Guión adaptado</strong>
          <span id="p5-info-out" class="badge badge-verde"></span>
        </div>
        <textarea id="p5-txt-salida" rows="9" style="background:#fbfefc;border-color:#86efac;font-size:0.93rem;line-height:1.6"></textarea>
        <div class="row" style="gap:6px;margin-top:10px">
          <button type="button" class="btn-primary btn-sm" id="btn-p5-copiar-out">Copiar</button>
          <button type="button" class="btn-outline btn-sm" id="btn-p5-descargar-out">Descargar TXT</button>
          <button type="button" class="btn-outline btn-sm" id="btn-p5-copiar-prompt" title="Copia un prompt con tu perfil de estilo y ejemplos reales">Copiar prompt con mi estilo</button>
        </div>
        <details class="plegable">
          <summary>Fragmentos tuyos usados como referencia <span class="hint">· RAG</span></summary>
          <div class="plegable-body">
            <div id="p5-rag-ejemplos" style="font-size:0.8rem;color:var(--text-2);display:flex;flex-direction:column;gap:6px"></div>
          </div>
        </details>
      </div>
    </div>

    <!-- Entrenar + perfil -->
    <div class="stack" style="gap:16px">
      <div class="card" style="margin:0">
        <span class="card-tag">Entrenar</span>
        <h2>Entrenar con mis transcripciones</h2>
        <p class="sub-desc">Elegí las sesiones y cuál hablante sos vos. Podés sumar archivos extra.</p>

        <div class="campo" style="margin-bottom:12px">
          <label>Sesiones guardadas</label>
          <div id="p5-lista-sesiones-train" style="display:flex;flex-direction:column;gap:6px;max-height:190px;overflow-y:auto;padding:8px;background:var(--surface-2);border:1px solid var(--border);border-radius:8px">
            <span style="font-size:0.82rem;color:var(--muted)">Cargando sesiones…</span>
          </div>
        </div>

        <div class="row" style="margin-bottom:14px">
          <input type="file" id="arch-p5-extra" accept=".txt,.srt,.vtt,.docx,.pdf" multiple hidden>
          <button type="button" class="btn-outline btn-sm" onclick="document.getElementById('arch-p5-extra').click()">+ Archivos extra</button>
          <span id="p5-lbl-extra" style="font-size:0.8rem;color:var(--muted)">Ningún archivo extra</span>
        </div>

        <button type="button" class="btn-primary btn-block" id="btn-p5-entrenar">Entrenar / actualizar modelo</button>
        <div id="p5-res-entrenar" style="margin-top:8px;font-size:0.84rem"></div>
      </div>

      <div class="card" id="p5-card-adn" style="margin:0">
        <div class="card-head" style="margin-bottom:12px">
          <div>
            <h2>Tu estilo aprendido</h2>
          </div>
          <a href="/api/estilo_dataset_jsonl" download="mi_estilo_finetuning.jsonl" class="btn-outline btn-xs" style="text-decoration:none;display:inline-flex;align-items:center;padding:4px 9px;border:1px solid var(--border-strong);border-radius:6px;font-size:0.75rem;font-weight:600;color:var(--text-2)" title="Pares guión neutro → tu estilo para fine-tuning">Dataset JSONL</a>
        </div>

        <div class="metricas" id="p5-metricas-grid">
          <div class="metrica"><div class="valor" id="p5-m-frases">0</div><div class="lbl">Frases analizadas</div></div>
          <div class="metrica"><div class="valor" id="p5-m-pal">0</div><div class="lbl">Palabras</div></div>
          <div class="metrica"><div class="valor" id="p5-m-long">0</div><div class="lbl">Palabras / oración</div></div>
        </div>

        <div style="margin-bottom:12px">
          <label class="lbl">Conectores y muletillas frecuentes</label>
          <div id="p5-chips-marcadores" class="chips"></div>
        </div>

        <div style="margin-bottom:12px">
          <label class="lbl" for="p5-inp-muletillas">Expresiones a usar al convertir <span style="font-weight:400;color:var(--muted)">(separadas por coma)</span></label>
          <div class="input-con-btn">
            <input type="text" id="p5-inp-muletillas" placeholder="vamos a ver, o sea, por ejemplo, fíjense que…">
            <button type="button" class="btn-outline btn-sm" id="btn-p5-guardar-muletillas">Guardar</button>
          </div>
        </div>

        <div>
          <label class="lbl">Expresiones más repetidas</label>
          <div id="p5-chips-ngramas" class="chips"></div>
        </div>
      </div>
    </div>
  </div>
</section>




</div><!-- /main-inner -->
</main>
</div><!-- /app -->
<script>
// ==============================================================================
// ESTADO GLOBAL DEL CLIENTE
// ==============================================================================
let colaP1 = [];
let sesionActiva = null;   // Objeto con la sesión actual cargada en Paso 3 y Paso 4
let grupoActivo = 'General';
let listaGrupos = [];
let nombresGrupoActual = [];
let archivoVideoP4 = null;

window.toggleGuiaBienvenida = function(forzar) {
  const box = document.getElementById('card-bienvenida-guia');
  if (!box) return;
  const abrir = (typeof forzar === 'boolean') ? forzar : !box.classList.contains('abierto');
  box.classList.toggle('abierto', abrir);
  document.body.style.overflow = abrir ? 'hidden' : '';
  try { localStorage.setItem('guia_vista_v2', '1'); } catch (e) {}
};
document.addEventListener('keydown', e => {
  if (e.key === 'Escape') toggleGuiaBienvenida(false);
});
// La guía se abre sola solo la primera vez
try {
  if (!localStorage.getItem('guia_vista_v2')) {
    window.addEventListener('load', () => toggleGuiaBienvenida(true));
  }
} catch (e) {}

function escHtml(str) {
  return String(str ?? '').replace(/[&<>"']/g, m => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[m]));
}

function fmtHMS(seg) {
  if (seg === null || seg === undefined || isNaN(seg)) return '--:--';
  const neg = seg < 0 ? '-' : '';
  seg = Math.abs(Number(seg));
  const total = Math.floor(seg);
  const dec = Math.round((seg - total) * 10);
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const s = total % 60;
  const base = h > 0
    ? `${String(h).padStart(2,'0')}:${String(m).padStart(2,'0')}:${String(s).padStart(2,'0')}`
    : `${String(m).padStart(2,'0')}:${String(s).padStart(2,'0')}`;
  return neg + (dec > 0 && dec < 10 ? `${base}.${dec}` : base);
}

function parseTiempoCliente(txt) {
  txt = String(txt || '').trim();
  if (!txt) return null;
  if (/^\d+(\.\d+)?$/.test(txt)) return parseFloat(txt);
  const p = txt.split(':');
  if (p.length === 2) return parseInt(p[0], 10) * 60 + parseFloat(p[1]);
  if (p.length === 3) return parseInt(p[0], 10) * 3600 + parseInt(p[1], 10) * 60 + parseFloat(p[2]);
  return null;
}

function fmtMB(bytes) {
  if (!bytes) return '';
  return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
}

// ==============================================================================
// NAVEGACIÓN ENTRE LOS 4 PASOS
// ==============================================================================
function irAPaso(num) {
  document.querySelectorAll('#stepper-nav .step-btn').forEach(b => {
    b.classList.toggle('activo', Number(b.dataset.step) === Number(num));
  });
  document.querySelectorAll('.step-section').forEach((sec, idx) => {
    sec.classList.toggle('activo', (idx + 1) === Number(num));
  });
  window.scrollTo({ top: 0, behavior: 'auto' });
  if (num === 2) {
    cargarAudiosDisponibles();
    cargarSesionesGuardadas();
    renderVinculoP2();
  }
  if (num === 3 && sesionActiva) {
    renderPaso3();
  }
  if (num === 4 && sesionActiva) {
    renderPaso4();
  }
  if (num === 5) {
    cargarEstadoEstiloPaso5();
  }
}

document.querySelectorAll('#stepper-nav .step-btn').forEach(b => {
  b.onclick = () => irAPaso(Number(b.dataset.step));
});

// ==============================================================================
// GITHUB SYNC
// ==============================================================================
async function consultarGitEstado(forzar = false) {
  try {
    const r = await fetch('/api/git_estado?verificar=' + (forzar ? '1' : '0'));
    if (!r.ok) return;
    const st = await r.json();
    const txt = document.getElementById('git-estado-txt');
    const btnCfg = document.getElementById('btn-git-cfg');
    if (!st.tiene_remote) {
      txt.innerHTML = '<span style="color:#d97706">Local (Sin remoto)</span>';
      btnCfg.style.display = 'inline-block';
      return;
    }
    btnCfg.style.display = 'none';
    if (st.estado === 'sincronizado') {
      txt.innerHTML = `<span style="color:#15803d">✔ Al día (${escHtml(st.rama)})</span>`;
    } else {
      txt.innerHTML = `<span>${escHtml(st.mensaje || st.estado)}</span>`;
    }
  } catch (e) {}
}
document.getElementById('btn-git-sinc').onclick = async () => {
  const r = await fetch('/api/git_sincronizar', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({accion:'auto'}) });
  if (r.ok) { const d = await r.json(); alert(d.mensaje || 'Sincronizado'); consultarGitEstado(); }
};
document.getElementById('btn-git-cfg').onclick = async () => {
  const url = prompt('Pegá la URL de tu repositorio en GitHub:');
  if (!url) return;
  await fetch('/api/git_sincronizar', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({accion:'configurar_remote', remote_url: url.trim()}) });
  consultarGitEstado(true);
};
consultarGitEstado(false);

// ==============================================================================
// PASO 1: CONVERTIR VIDEO / AUDIO A MP3 CON RECORTE POR MINUTO
// ==============================================================================
const zonaP1 = document.getElementById('zona-p1');
const archP1 = document.getElementById('arch-p1');
const btnP1Convertir = document.getElementById('btn-p1-convertir');

zonaP1.onclick = () => archP1.click();
zonaP1.ondragover = e => { e.preventDefault(); zonaP1.classList.add('sobre'); };
zonaP1.ondragleave = () => zonaP1.classList.remove('sobre');
zonaP1.ondrop = e => { e.preventDefault(); zonaP1.classList.remove('sobre'); agregarColaP1(e.dataTransfer.files); };
archP1.onchange = () => { agregarColaP1(archP1.files); archP1.value = ''; };

function agregarColaP1(files) {
  let primero = null;
  for (const f of files) {
    const it = { id: Math.random().toString(36).slice(2), file: f, nombre: f.name, bytes: f.size };
    colaP1.push(it);
    primero = primero || it;
  }
  renderColaP1();
  if (primero) verColaP1(primero.id);
}

function renderColaP1() {
  const box = document.getElementById('p1-cola-box');
  const lista = document.getElementById('p1-cola-lista');
  const tit = document.getElementById('p1-cola-titulo');
  if (!colaP1.length) {
    box.style.display = 'none';
    btnP1Convertir.disabled = true;
    return;
  }
  box.style.display = 'block';
  btnP1Convertir.disabled = false;
  tit.textContent = `Archivos cargados (${colaP1.length}) · podés procesarlos varias veces con distintas opciones`;
  actualizarPreviewSplitP1();
  lista.innerHTML = colaP1.map(it => `
    <li class="cola-item">
      <span><strong>${escHtml(it.nombre)}</strong> <small style="color:var(--muted)">(${fmtMB(it.bytes)})</small></span>
      <span class="row" style="gap:4px;flex-wrap:nowrap">
        <button type="button" class="btn-outline btn-xs" onclick="verColaP1('${it.id}')">Ver</button>
        <button type="button" class="btn-danger-soft btn-xs" onclick="quitarColaP1('${it.id}')">✕ Quitar</button>
      </span>
    </li>
  `).join('');
}

window.quitarColaP1 = function(id) {
  colaP1 = colaP1.filter(x => x.id !== id);
  renderColaP1();
  if (prevP1.origen === 'cola:' + id) {
    if (colaP1.length) verColaP1(colaP1[0].id); else ocultarPreviewP1();
  }
};
document.getElementById('btn-p1-limpiar').onclick = () => {
  colaP1 = []; renderColaP1();
  if (prevP1.origen.startsWith('cola:')) ocultarPreviewP1();
};

btnP1Convertir.onclick = async () => {
  if (!colaP1.length) return;
  if (tipoCorteP1() === 'partes') {
    const err = validarSplitP1();
    if (err) { alert(err); return; }
  }
  // El archivo queda en la lista para poder procesarlo de nuevo con otras opciones
  const lotes = [...colaP1];
  btnP1Convertir.disabled = true;
  btnP1Convertir.textContent = 'Procesando…';
  try {
    await ejecutarLotePaso1(lotes, false);
  } finally {
    actualizarUiCorteP1();
    btnP1Convertir.disabled = !colaP1.length;
  }
};

async function ejecutarLotePaso1(lotes, usarExistente = false) {
  const params = {
    cal: document.getElementById('p1-cal').value,
    desde: document.getElementById('p1-desde').value.trim(),
    hasta: document.getElementById('p1-hasta').value.trim(),
    mb: document.getElementById('p1-mb').value.trim(),
    mono: document.getElementById('p1-mono').checked ? '1' : '0',
    modo_p1: document.getElementById('p1-modo-salida').value,
    solo_extraer: '1'
  };
  if (tipoCorteP1() === 'partes') {
    const sp = leerSplitP1();
    params.split_modo = sp.modo;
    params.split_puntos = sp.puntos;
    params.split_duracion = sp.duracion;
    params.split_exacto = document.getElementById('p1-split-exacto').checked ? '1' : '0';
  }

  for (const item of lotes) {
    const pBox = document.createElement('div');
    pBox.className = 'progreso-item';
    pBox.innerHTML = `
      <div style="display:flex;justify-content:space-between;font-weight:700;font-size:0.9rem">
        <span>⏳ Procesando ${escHtml(item.nombre)}… <span class="msg"></span></span>
        <span class="pct">10%</span>
      </div>
      <div class="barra-fondo"><div class="barra-relleno" style="width:10%"></div></div>
    `;
    document.getElementById('p1-progreso').prepend(pBox);

    try {
      const q = new URLSearchParams({ ...params, nombre: item.nombre, usar_mp3_existente: usarExistente ? '1' : '0' });
      const r = await fetch('/api/iniciar?' + q.toString(), { method: 'POST', body: usarExistente ? null : item.file });
      if (!r.ok) throw new Error(await r.text());
      const { job_id } = await r.json();

      while (true) {
        await new Promise(res => setTimeout(res, 700));
        const st = await fetch('/api/estado?id=' + encodeURIComponent(job_id));
        const d = await st.json();
        pBox.querySelector('.msg').textContent = d.mensaje || '';
        pBox.querySelector('.pct').textContent = (d.progreso || 0) + '%';
        pBox.querySelector('.barra-relleno').style.width = (d.progreso || 0) + '%';
        if (d.estado === 'error') throw new Error(d.error || 'Error');
        if (d.estado === 'completado' && d.resultado && d.resultado.modo === 'partes') {
          await cargarAudiosDisponibles();
          pBox.innerHTML = renderPartesP1(item.nombre, d.resultado.partes || []);
          break;
        }
        if (d.estado === 'completado') {
          const nomMp3 = d.resultado.nombre_mp3 || '';
          const nomVid = d.resultado.nombre_video || '';
          await cargarAudiosDisponibles(nomMp3 || nomVid);
          let htmlOut = '<div style="display:flex;flex-direction:column;gap:8px">';
          if (nomVid) {
            htmlOut += `
              <div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:10px">
                <div><strong style="color:var(--ok)">🎬 Video MP4 recortado listo:</strong> ${escHtml(nomVid)}</div>
                <div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap">
                  <a class="btn-primary btn-sm" style="text-decoration:none" href="/audios/${encodeURIComponent(nomVid)}" download>⬇ Descargar Video MP4</a>
                </div>
              </div>
            `;
          }
          if (nomMp3) {
            htmlOut += `
              <div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:10px">
                <div><strong style="color:var(--ok)">🎵 Audio MP3 listo:</strong> ${escHtml(nomMp3)}</div>
                <div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap">
                  <audio controls src="/audios/${encodeURIComponent(nomMp3)}" style="height:34px"></audio>
                  <a class="btn-outline btn-sm" style="text-decoration:none" href="/audios/${encodeURIComponent(nomMp3)}" download>⬇ Descargar MP3</a>
                  <button type="button" class="btn-primary btn-sm" onclick="seleccionarMp3EIrAPaso2('${escHtml(nomMp3)}')">
                    👉 Seguir al Paso 2 (Transcribir este MP3)
                  </button>
                </div>
              </div>
            `;
          }
          htmlOut += '</div>';
          pBox.innerHTML = htmlOut;
          break;
        }
      }
    } catch (e) {
      pBox.innerHTML = `<div style="color:var(--danger);font-weight:700">✘ Error: ${escHtml(e.message)}</div>`;
    }
  }
}

document.getElementById('btn-p1-recortar-existente').onclick = async () => {
  const nom = document.getElementById('p1-sel-existente').value;
  if (!nom) { alert('Elegí un archivo de la lista.'); return; }
  const d = document.getElementById('p1-desde').value.trim();
  const h = document.getElementById('p1-hasta').value.trim();
  if (tipoCorteP1() === 'partes') {
    const err = validarSplitP1();
    if (err) { alert(err); return; }
  } else if (!d && !h) { alert('Escribí el minuto:segundo en "Desde" y/o "Hasta" arriba para recortar.'); return; }
  const btn = document.getElementById('btn-p1-recortar-existente');
  btn.disabled = true;
  try {
    await ejecutarLotePaso1([{ nombre: nom }], true);
  } finally {
    btn.disabled = false;
  }
};


// ==============================================================================
// PASO 4 · RESULTADOS: VIDEO (silenciar / subtitular voces) + reproductor común
// ==============================================================================
const pub = { video: '', sesion: null, modo: {}, nombres: {}, idxMuestra: {}, miVoz: null };

document.querySelectorAll('input[name="pub-nombres"]').forEach(r => r.addEventListener('change', () => {
  document.getElementById('pub-box-prefijo').style.display = pubModoNombre() === 'anon' ? 'block' : 'none';
  renderVocesPub();
}));
document.getElementById('pub-prefijo').addEventListener('input', renderEtiquetasPub);
['pub-igualar', 'pub-nivelar'].forEach(id => document.getElementById(id).addEventListener('change', renderEtiquetasPub));
function pubModoNombre() { return document.querySelector('input[name="pub-nombres"]:checked').value; }

// Vista previa del video
function cargarPreviewPub(nombre) {
  const v = document.getElementById('pub-audio');
  if (!nombre) { if (v.getAttribute('src')) { v.removeAttribute('src'); v.load(); } return; }
  const src = '/audios/' + encodeURIComponent(nombre);
  if (v.getAttribute('src') !== src) { v.src = src; }
}
function offsetPub() { return sesionActiva ? (Number(sesionActiva.offset) || 0) : 0; }
function segActualPub(t) {
  if (!pub.sesion) return null;
  const off = offsetPub();
  let mejor = null;
  for (const s of pub.sesion.segmentos) {
    if (s.inicio + off <= t && t < s.fin + off) { if (!mejor || modoPub(s.hablante) !== 'normal') mejor = s; }
  }
  return mejor;
}
function actualizarVivoPub() {
  const v = document.getElementById('pub-audio');
  const t = v.currentTime || 0;
  document.getElementById('pub-prev-t').textContent = fmtCorte(t);
  const dur = isFinite(v.duration) ? v.duration : 0;
  document.getElementById('pub-tl-cabezal').style.left = dur ? `calc(${Math.min(100, t / dur * 100)}% - 1px)` : '0';
  const sub = document.getElementById('pub-sub-live');
  const tag = document.getElementById('pub-mute-tag');
  const info = document.getElementById('pub-audio-info');
  const seg = segActualPub(t);
  const simular = pub.sesion && document.getElementById('pub-simular').checked;
  if (!pub.sesion) return;
  if (!seg) {
    if (!pub._escuchando) info.textContent = 'Nadie habla en este momento.';
    sub.style.display = 'none'; tag.style.display = 'none';
    if (simular) v.muted = false;
    return;
  }
  const h = pub.sesion.hablantes.find(x => x.id === seg.hablante);
  const nombre = pub.nombres[seg.hablante] || h?.nombre || seg.hablante;
  const m = modoPub(seg.hablante);
  const etiq = etiquetasPub()[seg.hablante];
  const textoModo = { normal: 'se escucha', silenciar: 'silenciada + subtítulo', subtitulo: 'solo subtítulo' }[m];
  if (!pub._escuchando) info.innerHTML = `Ahora habla <strong>${escHtml(nombre)}</strong> · ${textoModo}`;
  if (simular && m !== 'normal') {
    const pref = pubModoNombre() === 'ninguno' ? '' : `[${etiq}] `;
    sub.textContent = pref + (seg.texto || '');
    sub.style.display = 'inline-block';
  } else {
    sub.style.display = 'none';
  }
  // Al tocar ▶ Escuchar en una voz se la oye aunque esté marcada para silenciar
  const silenciada = simular && m === 'silenciar' && !pub._escuchando;
  v.muted = !!silenciada;
  tag.style.display = silenciada ? 'inline-block' : 'none';
}
function renderTimelinePub() {
  const capa = document.getElementById('pub-tl-marcas');
  const v = document.getElementById('pub-audio');
  const dur = isFinite(v.duration) ? v.duration : (pub.sesion?.duracion || 0);
  if (!capa || !dur) return;
  document.getElementById('pub-prev-dur').textContent = fmtCorte(dur);
  if (!pub.sesion) { capa.innerHTML = ''; return; }
  const off = offsetPub();
  const pct = t => Math.max(0, Math.min(100, t / dur * 100));
  let html = '';
  for (const s of pub.sesion.segmentos) {
    const m = modoPub(s.hablante);
    if (m === 'normal') continue;
    const a = pct(s.inicio + off), b = pct(s.fin + off);
    if (b <= a) continue;
    html += `<div class="tl-seg ${m === 'silenciar' ? 'sil' : 'sub'}" style="left:${a}%;width:${b - a}%"></div>`;
  }
  capa.innerHTML = html;
  document.getElementById('pub-tl-leyenda').style.display = html ? 'flex' : 'none';
}
(function () {
  const v = document.getElementById('pub-audio');
  v.addEventListener('timeupdate', actualizarVivoPub);
  v.addEventListener('seeked', actualizarVivoPub);
  v.addEventListener('loadedmetadata', () => { renderTimelinePub(); actualizarVivoPub(); });
  v.addEventListener('pause', () => { pub._escuchando = false; });
  document.getElementById('pub-timeline').addEventListener('click', e => {
    const dur = isFinite(v.duration) ? v.duration : 0;
    if (!dur) return;
    const r = e.currentTarget.getBoundingClientRect();
    v.currentTime = Math.max(0, Math.min(dur, (e.clientX - r.left) / r.width * dur));
  });
  document.querySelectorAll('[data-salto-pub]').forEach(b => b.onclick = () => {
    v.currentTime = Math.max(0, (v.currentTime || 0) + Number(b.dataset.saltoPub));
  });
  document.getElementById('pub-simular').addEventListener('change', () => { if (!document.getElementById('pub-simular').checked) v.muted = false; actualizarVivoPub(); });
})();

function barraProgreso(cont, texto) {
  const box = document.createElement('div');
  box.className = 'progreso-item';
  box.innerHTML = `<div style="display:flex;justify-content:space-between;font-weight:600;font-size:0.86rem">
      <span>${escHtml(texto)} <span class="msg" style="font-weight:400;color:var(--muted)"></span></span><span class="pct">0%</span></div>
    <div class="barra-fondo"><div class="barra-relleno"></div></div>`;
  cont.innerHTML = '';
  cont.appendChild(box);
  return box;
}
async function seguirJob(jobId, box) {
  while (true) {
    await new Promise(res => setTimeout(res, 800));
    const d = await (await fetch('/api/estado?id=' + encodeURIComponent(jobId))).json();
    box.querySelector('.msg').textContent = d.mensaje || '';
    box.querySelector('.pct').textContent = (d.progreso || 0) + '%';
    box.querySelector('.barra-relleno').style.width = (d.progreso || 0) + '%';
    if (d.estado === 'error') throw new Error(d.error || 'Error');
    if (d.estado === 'completado') return d.resultado;
  }
}

function vocesOrdenadas() {
  return [...(pub.sesion?.hablantes || [])].sort((a, b) => (b.duracion || 0) - (a.duracion || 0));
}
function modoPub(spk) { return pub.modo[spk] || 'normal'; }
function subtituladosPub() {
  return new Set(Object.keys(pub.modo).filter(k => pub.modo[k] === 'silenciar' || pub.modo[k] === 'subtitulo'));
}
function etiquetasPub() {
  const modo = pubModoNombre();
  const pref = document.getElementById('pub-prefijo').value.trim() || 'Alumno';
  const subs = subtituladosPub();
  const orden = [];
  [...(pub.sesion?.segmentos || [])].sort((a, b) => a.inicio - b.inicio).forEach(s => {
    if (subs.has(s.hablante) && !orden.includes(s.hablante)) orden.push(s.hablante);
  });
  const m = {};
  orden.forEach((spk, i) => {
    const h = pub.sesion.hablantes.find(x => x.id === spk);
    m[spk] = modo === 'anon' ? `${pref} ${i + 1}` : modo === 'real' ? (pub.nombres[spk] || h?.nombre || spk) : '(sin nombre)';
  });
  return m;
}
function renderEtiquetasPub() {
  renderTimelinePub();
  if (document.getElementById('pub-audio').readyState > 0) actualizarVivoPub();
  const et = etiquetasPub();
  document.querySelectorAll('#pub-lista-voces .voz-fila').forEach(f => {
    const spk = f.dataset.spk, m = modoPub(spk);
    f.classList.toggle('silenciada', m === 'silenciar');
    f.classList.toggle('subtitulo', m === 'subtitulo');
    const e = f.querySelector('.etq');
    e.classList.toggle('azul', m === 'subtitulo');
    e.textContent = et[spk] ? (m === 'silenciar' ? `→ silenciada, subtítulo como "${et[spk]}"` : `→ se escucha + subtítulo como "${et[spk]}"`) : '';
  });
  const segs = pub.sesion?.segmentos || [];
  const nSil = Object.values(pub.modo).filter(x => x === 'silenciar').length;
  const nSub = Object.values(pub.modo).filter(x => x === 'subtitulo').length;
  const subs = subtituladosPub();
  const nLineas = segs.filter(s => subs.has(s.hablante)).length;
  const partes = [];
  if (nSil) partes.push(`${nSil} ${nSil === 1 ? 'voz silenciada' : 'voces silenciadas'}`);
  if (nSub) partes.push(`${nSub} ${nSub === 1 ? 'voz' : 'voces'} solo con subtítulo`);
  document.getElementById('pub-resumen-sel').textContent = partes.length
    ? `${partes.join(' · ')} · ${nLineas} intervenciones subtituladas`
    : 'Todavía no elegiste ninguna voz para silenciar o subtitular.';
  const soloAudio = document.getElementById('pub-igualar').checked || document.getElementById('pub-nivelar').checked;
  document.getElementById('btn-pub-generar').disabled = !subs.size && !soloAudio;
  if (!partes.length && soloAudio) document.getElementById('pub-resumen-sel').textContent = 'Sin subtítulos: solo se procesa el volumen.';
}
function renderVocesPub() {
  const cont = document.getElementById('pub-lista-voces');
  if (!pub.sesion) { cont.innerHTML = ''; return; }
  const voces = vocesOrdenadas();
  const opciones = [['normal', 'Se escucha'], ['silenciar', 'Silenciar'], ['subtitulo', 'Solo subtítulo']];
  cont.innerHTML = voces.map((h, i) => `
    <div class="voz-fila" data-spk="${escHtml(h.id)}">
      <span class="dot" style="background:${escHtml(h.color || '#94a3b8')}"></span>
      <span class="voz-nom">${escHtml(h.nombre ?? h.id)}</span>
      <div class="voz-stats">${h.cantidad || 0} ${h.cantidad === 1 ? 'intervención' : 'intervenciones'} · ${fmtHMS(h.duracion || 0)}<br><span class="etq"></span></div>
      <div class="row" style="gap:4px;flex-wrap:nowrap">
        <button type="button" class="btn-outline btn-xs" data-play="${escHtml(h.id)}">▶ Escuchar</button>
        <button type="button" class="btn-yo ${pub.miVoz === h.id ? 'activo' : ''}" data-yo="${escHtml(h.id)}">${pub.miVoz === h.id ? '★ Soy yo' : '☆ Soy yo'}</button>
      </div>
      <div class="seg3" role="radiogroup" aria-label="Qué hacer con esta voz">
        ${opciones.map(([v, t]) => `<label><input type="radio" name="pubm-${escHtml(h.id)}" value="${v}" data-spk="${escHtml(h.id)}" ${modoPub(h.id) === v ? 'checked' : ''}><span>${t}</span></label>`).join('')}
      </div>
    </div>`).join('');
  cont.querySelectorAll('.seg3 input').forEach(r => r.onchange = () => { pub.modo[r.dataset.spk] = r.value; renderEtiquetasPub(); });
  cont.querySelectorAll('.pub-inp-nombre').forEach(inp => inp.oninput = () => { pub.nombres[inp.dataset.spk] = inp.value; renderEtiquetasPub(); });
  cont.querySelectorAll('[data-play]').forEach(b => b.onclick = () => escucharVozPub(b.dataset.play));
  cont.querySelectorAll('[data-yo]').forEach(b => b.onclick = () => {
    pub.miVoz = b.dataset.yo;
    if (sesionActiva) sesionActiva.miVozId = pub.miVoz;
    const h = pub.sesion.hablantes.find(x => x.id === pub.miVoz);
    recordarMiVoz(pub.sesion.sesion_id, h || null);
    renderVocesPub();
    actualizarResumenAudioPaso3();
  });
  renderEtiquetasPub();
}
function escucharVozPub(spk) {
  const segs = (pub.sesion.segmentos || []).filter(s => s.hablante === spk && (s.fin - s.inicio) >= 1.5);
  const lista = segs.length ? segs : (pub.sesion.segmentos || []).filter(s => s.hablante === spk);
  if (!lista.length) return;
  const k = Math.min(lista.length, 8);
  const i = (pub.idxMuestra[spk] || 0) % k;
  pub.idxMuestra[spk] = i + 1;
  const s = lista[Math.floor(i * lista.length / k)];
  const audio = document.getElementById('pub-audio');
  const h = pub.sesion.hablantes.find(x => x.id === spk);
  document.getElementById('pub-audio-info').innerHTML =
    `<strong>${escHtml(pub.nombres[spk] || h?.nombre || spk)}</strong> · ${fmtHMS(s.inicio)} · “${escHtml((s.texto || '').slice(0, 90))}”`;
  pub._escuchando = true;
  audio.currentTime = Math.max(0, s.inicio + offsetPub());
  audio.play().catch(() => {});
  clearTimeout(pub._stop);
  pub._stop = setTimeout(() => audio.pause(), Math.min(8, s.fin - s.inicio + 0.3) * 1000);
}
document.querySelectorAll('[data-bulk]').forEach(b => b.onclick = () => {
  const [quien, accion] = b.dataset.bulk.split(':');
  const v = vocesOrdenadas();
  const objetivo = quien === 'demas' ? v.filter(h => h.id !== pub.miVoz) : v;
  if (quien === 'demas' && pub.miVoz) pub.modo[pub.miVoz] = 'normal';
  document.querySelectorAll('.receta').forEach(x => x.classList.remove('activa'));
  objetivo.forEach(h => { pub.modo[h.id] = accion; });
  renderVocesPub();
});

function aplicarRecetaPub(receta) {
  if (!pub.sesion) return;
  const v = vocesOrdenadas();
  const demas = v.filter(h => h.id !== pub.miVoz);
  const nombres = valor => { document.querySelector(`input[name="pub-nombres"][value="${valor}"]`).checked = true;
    document.getElementById('pub-box-prefijo').style.display = valor === 'anon' ? 'block' : 'none'; };
  pub.modo = {};
  if (receta === 'publicar') { demas.forEach(h => pub.modo[h.id] = 'silenciar'); nombres('anon'); }
  if (receta === 'sub-alumnos') { demas.forEach(h => pub.modo[h.id] = 'subtitulo'); nombres('real'); }
  if (receta === 'sub-todos') { v.forEach(h => pub.modo[h.id] = 'subtitulo'); nombres('real'); }
  document.getElementById('pub-igualar').checked = receta !== 'sub-todos';
  document.getElementById('pub-nivelar').checked = receta !== 'sub-todos';
  document.querySelectorAll('.receta').forEach(x => x.classList.toggle('activa', x.dataset.recetaPub === receta));
  renderVocesPub();
}
document.querySelectorAll('[data-receta-pub]').forEach(b => b.onclick = () => aplicarRecetaPub(b.dataset.recetaPub));

document.getElementById('btn-pub-generar').onclick = async () => {
  if (!sesionActiva || !sesionActiva.sincronizado) { alert('Primero sincronizá la transcripción con su video (paso 2).'); return; }
  const igualarPub = document.getElementById('pub-igualar').checked;
  if (!pub.sesion || (!subtituladosPub().size && !igualarPub && !document.getElementById('pub-nivelar').checked)) return;
  const cont = document.getElementById('pub-progreso-render');
  const btn = document.getElementById('btn-pub-generar');
  btn.disabled = true;
  const box = barraProgreso(cont, 'Generando video');
  try {
    const r = await fetch('/api/publicar_render', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        sesion_id: pub.sesion.sesion_id,
        nombre_video: pub.video,
        silenciados: Object.keys(pub.modo).filter(k => pub.modo[k] === 'silenciar'),
        subtitular: Object.keys(pub.modo).filter(k => pub.modo[k] === 'subtitulo'),
        offset: offsetPub(),
        nombres: pub.nombres,
        modo_nombre: pubModoNombre(),
        prefijo_anon: document.getElementById('pub-prefijo').value.trim() || 'Alumno',
        nivelar: document.getElementById('pub-nivelar').checked,
        igualar: igualarPub,
        margen: document.getElementById('pub-margen').checked ? 0.2 : 0
      })
    });
    if (!r.ok) throw new Error(await r.text());
    const res = await seguirJob((await r.json()).job_id, box);
    const urlV = '/audios/' + encodeURIComponent(res.nombre_video);
    const urlS = '/audios/' + encodeURIComponent(res.nombre_srt);
    cont.innerHTML = `<div class="progreso-item pub-resultado">
      <div class="row-between" style="margin-bottom:10px">
        <strong style="color:var(--ok)">✔ Video listo: ${escHtml(res.nombre_video)}</strong>
        <div class="row" style="gap:6px">
          <a class="btn-primary btn-sm" style="text-decoration:none" href="${urlV}" download>Descargar MP4</a>
          ${res.nombre_srt ? `<a class="btn-outline btn-sm" style="text-decoration:none" href="${urlS}" download>Descargar SRT</a>` : ''}
        </div>
      </div>
      <video id="pub-video-res" controls preload="metadata" src="${urlV}"></video>
      <div style="font-size:0.8rem;color:var(--muted);margin-top:8px">
        ${res.tramos_silenciados ? `${res.tramos_silenciados} tramos silenciados (${fmtHMS(res.segundos_silenciados)}) · ` : ''}${res.subtitulos} subtítulos${res.igualado ? ' · voces igualadas' : ''}${res.nivelado ? ' · audio nivelado' : ''}.
        ${res.nombre_srt ? 'El MP4 lleva los subtítulos como pista interna; si la plataforma no los muestra, subí también el SRT.' : ''}
      </div>
    </div>`;
    // Vista previa con los subtítulos superpuestos (convierte el SRT a VTT en el navegador)
    if (res.nombre_srt) try {
      const srt = await (await fetch(urlS)).text();
      const vtt = 'WEBVTT\n\n' + srt.replace(/\r/g, '').replace(/(\d{2}:\d{2}:\d{2}),(\d{3})/g, '$1.$2');
      const tr = document.createElement('track');
      tr.kind = 'subtitles'; tr.label = 'Subtítulos'; tr.srclang = 'es'; tr.default = true;
      tr.src = URL.createObjectURL(new Blob([vtt], { type: 'text/vtt' }));
      document.getElementById('pub-video-res').appendChild(tr);
    } catch (e) {}
    cargarAudiosDisponibles();
  } catch (e) {
    cont.innerHTML = `<div style="color:var(--danger);font-weight:600;margin-top:10px">✘ ${escHtml(e.message)}</div>`;
  } finally {
    btn.disabled = false;
  }
};

// ---------- Vista previa con minuto:segundo (paso 1) ----------
const prevP1 = { origen: '', dur: 0, el: null };
const EXT_VIDEO_P1 = /\.(mp4|mkv|mov|avi|webm)$/i;

function fmtCorte(seg) {
  seg = Math.max(0, Math.floor(seg || 0));
  const h = Math.floor(seg / 3600), m = Math.floor((seg % 3600) / 60), s = seg % 60;
  return h ? `${h}:${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`
           : `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
}

function cargarPreviewP1(src, nombre, esVideo, origen) {
  const box = document.getElementById('p1-preview');
  const v = document.getElementById('p1-prev-video');
  const a = document.getElementById('p1-prev-audio');
  [v, a].forEach(x => { x.pause(); x.removeAttribute('src'); x.load(); });
  prevP1.origen = origen;
  prevP1.dur = 0;
  prevP1.el = esVideo ? v : a;
  v.style.display = esVideo ? 'block' : 'none';
  a.style.display = esVideo ? 'none' : 'block';
  document.getElementById('p1-prev-error').style.display = 'none';
  document.getElementById('p1-prev-nombre').textContent = nombre;
  document.getElementById('p1-prev-tipo').textContent = esVideo ? 'Video' : 'Audio';
  document.getElementById('p1-prev-t').textContent = '00:00';
  document.getElementById('p1-prev-dur').textContent = '--:--';
  prevP1.el.src = src;
  box.style.display = 'block';
  document.getElementById('zona-p1').classList.add('con-archivo');
  document.getElementById('zona-p1-titulo').textContent = 'Agregar otro video o audio';
  renderTimelineP1();
}
function ocultarPreviewP1() {
  const v = document.getElementById('p1-prev-video'), a = document.getElementById('p1-prev-audio');
  [v, a].forEach(x => { x.pause(); x.removeAttribute('src'); x.load(); });
  prevP1.origen = ''; prevP1.dur = 0; prevP1.el = null;
  document.getElementById('p1-preview').style.display = 'none';
  document.getElementById('zona-p1').classList.remove('con-archivo');
  document.getElementById('zona-p1-titulo').textContent = 'Arrastrá acá tu video o audio';
}
window.verColaP1 = function(id) {
  const it = colaP1.find(x => x.id === id);
  if (!it) return;
  if (!it.url) it.url = URL.createObjectURL(it.file);
  const esVideo = (it.file.type || '').startsWith('video') || EXT_VIDEO_P1.test(it.nombre);
  cargarPreviewP1(it.url, it.nombre, esVideo, 'cola:' + id);
};
document.getElementById('btn-p1-ver-existente').onclick = () => {
  const sel = document.getElementById('p1-sel-existente');
  if (!sel.value) return;
  cargarPreviewP1('/audios/' + encodeURIComponent(sel.value), sel.value, EXT_VIDEO_P1.test(sel.value), 'audios:' + sel.value);
  document.getElementById('p1-preview').scrollIntoView({ behavior: 'smooth', block: 'center' });
};

['p1-prev-video', 'p1-prev-audio'].forEach(id => {
  const el = document.getElementById(id);
  el.addEventListener('loadedmetadata', () => {
    if (el !== prevP1.el) return;
    prevP1.dur = isFinite(el.duration) ? el.duration : 0;
    document.getElementById('p1-prev-dur').textContent = prevP1.dur ? fmtCorte(prevP1.dur) : '--:--';
    renderTimelineP1();
  });
  el.addEventListener('timeupdate', () => { if (el === prevP1.el) actualizarCabezalP1(); });
  el.addEventListener('seeked', () => { if (el === prevP1.el) actualizarCabezalP1(); });
  el.addEventListener('error', () => {
    if (el !== prevP1.el || !el.getAttribute('src')) return;
    document.getElementById('p1-prev-error').style.display = 'block';
  });
});

function actualizarCabezalP1() {
  const el = prevP1.el;
  if (!el) return;
  document.getElementById('p1-prev-t').textContent = fmtCorte(el.currentTime);
  const pct = prevP1.dur ? Math.min(100, el.currentTime / prevP1.dur * 100) : 0;
  document.getElementById('p1-tl-cabezal').style.left = `calc(${pct}% - 1px)`;
}

function renderTimelineP1() {
  const capa = document.getElementById('p1-tl-marcas');
  const partes = tipoCorteP1() === 'partes';
  document.getElementById('p1-prev-acciones-tramo').style.display = partes ? 'none' : 'flex';
  document.getElementById('p1-prev-acciones-partes').style.display = partes ? 'flex' : 'none';
  const dur = prevP1.dur;
  if (!capa || !dur) { if (capa) capa.innerHTML = ''; actualizarCabezalP1(); return; }
  const pct = t => `${Math.max(0, Math.min(100, t / dur * 100))}%`;
  let html = '';
  if (partes) {
    const lista = calcularPartesCliente(dur);
    lista.forEach(([a, b], i) => {
      html += `<div class="tl-parte ${i % 2 ? 'impar' : 'par'}" style="left:${pct(a)};width:calc(${pct(b)} - ${pct(a)})" title="Parte ${i + 1}: ${fmtCorte(a)} → ${fmtCorte(b)}">${i + 1}</div>`;
      if (i > 0) html += `<div class="tl-corte" style="left:calc(${pct(a)} - 1px)"></div>`;
    });
  } else {
    const d = parseTiempoCliente(document.getElementById('p1-desde').value);
    const h = parseTiempoCliente(document.getElementById('p1-hasta').value);
    if ((d !== null && !isNaN(d)) || (h !== null && !isNaN(h))) {
      const a = (d !== null && !isNaN(d)) ? d : 0;
      const b = (h !== null && !isNaN(h)) ? Math.min(h, dur) : dur;
      if (b > a) html += `<div class="tl-rango" style="left:${pct(a)};width:calc(${pct(b)} - ${pct(a)})" title="${fmtCorte(a)} → ${fmtCorte(b)}"></div>`;
    }
  }
  capa.innerHTML = html;
  actualizarCabezalP1();
}

document.getElementById('p1-timeline').addEventListener('click', e => {
  if (!prevP1.el || !prevP1.dur) return;
  const r = e.currentTarget.getBoundingClientRect();
  prevP1.el.currentTime = Math.max(0, Math.min(prevP1.dur, (e.clientX - r.left) / r.width * prevP1.dur));
});
document.querySelectorAll('[data-salto]').forEach(b => b.onclick = () => {
  if (!prevP1.el) return;
  prevP1.el.currentTime = Math.max(0, Math.min(prevP1.dur || Infinity, prevP1.el.currentTime + Number(b.dataset.salto)));
});
function tiempoActualP1() { return prevP1.el ? prevP1.el.currentTime : 0; }
document.getElementById('btn-p1-marcar-desde').onclick = () => {
  document.getElementById('p1-desde').value = fmtCorte(tiempoActualP1());
  renderTimelineP1(); actualizarPreviewSplitP1();
};
document.getElementById('btn-p1-marcar-hasta').onclick = () => {
  document.getElementById('p1-hasta').value = fmtCorte(Math.ceil(tiempoActualP1()));
  renderTimelineP1(); actualizarPreviewSplitP1();
};
document.getElementById('btn-p1-agregar-corte').onclick = () => {
  document.querySelector('input[name="p1-split-modo"][value="puntos"]').checked = true;
  const inp = document.getElementById('p1-split-puntos');
  const actuales = inp.value.split(/[\s,;]+/).filter(Boolean);
  const nuevo = fmtCorte(tiempoActualP1());
  if (!actuales.includes(nuevo)) actuales.push(nuevo);
  actuales.sort((x, y) => (parseTiempoCliente(x) || 0) - (parseTiempoCliente(y) || 0));
  inp.value = actuales.join(', ');
  renderTimelineP1(); actualizarPreviewSplitP1();
};
document.getElementById('btn-p1-quitar-corte').onclick = () => {
  const inp = document.getElementById('p1-split-puntos');
  const actuales = inp.value.split(/[\s,;]+/).filter(Boolean);
  actuales.pop();
  inp.value = actuales.join(', ');
  renderTimelineP1(); actualizarPreviewSplitP1();
};

// ---------- División en partes (split) ----------
function tipoCorteP1() {
  const r = document.querySelector('input[name="p1-tipo-corte"]:checked');
  return r ? r.value : 'tramo';
}
function leerSplitP1() {
  const r = document.querySelector('input[name="p1-split-modo"]:checked');
  return {
    modo: r ? r.value : 'puntos',
    puntos: document.getElementById('p1-split-puntos').value.trim(),
    duracion: document.getElementById('p1-split-duracion').value.trim()
  };
}
function validarSplitP1() {
  const sp = leerSplitP1();
  if (sp.modo === 'puntos') {
    const pts = sp.puntos.split(/[\s,;]+/).filter(Boolean);
    if (!pts.length) return 'Escribí al menos un punto de corte, por ejemplo: 10:00, 25:30';
    const malo = pts.find(t => parseTiempoCliente(t) === null || isNaN(parseTiempoCliente(t)));
    if (malo) return `"${malo}" no es un tiempo válido (usá mm:ss o hh:mm:ss).`;
  } else {
    const dur = parseTiempoCliente(sp.duracion);
    if (!dur || dur < 1) return 'Escribí la duración de cada parte, por ejemplo: 10:00';
  }
  return '';
}

// Cálculo local de las partes, igual al del servidor, para la vista previa
function calcularPartesCliente(duracion) {
  const sp = leerSplitP1();
  let ini = parseTiempoCliente(document.getElementById('p1-desde').value) || 0;
  let fin = parseTiempoCliente(document.getElementById('p1-hasta').value);
  if (fin === null || isNaN(fin) || fin > duracion) fin = duracion;
  if (fin - ini <= 0.5) return [];
  let bordes;
  if (sp.modo === 'puntos') {
    const pts = [...new Set(sp.puntos.split(/[\s,;]+/).filter(Boolean).map(parseTiempoCliente))]
      .filter(t => t !== null && !isNaN(t) && t > ini + 0.5 && t < fin - 0.5).sort((a, b) => a - b);
    if (!pts.length) return [];
    bordes = [ini, ...pts, fin];
  } else {
    const paso = parseTiempoCliente(sp.duracion);
    if (!paso || paso < 1 || (fin - ini) / paso > 500) return [];
    bordes = [ini];
    while (bordes[bordes.length - 1] + paso < fin - 0.5) bordes.push(bordes[bordes.length - 1] + paso);
    bordes.push(fin);
  }
  return bordes.slice(0, -1).map((a, i) => [a, bordes[i + 1]]);
}

const cacheDuracionP1 = new Map();
function duracionLocal(src, clave) {
  if (cacheDuracionP1.has(clave)) return Promise.resolve(cacheDuracionP1.get(clave));
  return new Promise(res => {
    const v = document.createElement('video');
    v.preload = 'metadata';
    const fin = val => { cacheDuracionP1.set(clave, val); res(val); };
    v.onloadedmetadata = () => fin(isFinite(v.duration) ? v.duration : null);
    v.onerror = () => fin(null);
    v.src = src;
    setTimeout(() => res(cacheDuracionP1.get(clave) ?? null), 6000);
  });
}

let _previewTimerP1 = null;
function actualizarPreviewSplitP1() {
  clearTimeout(_previewTimerP1);
  _previewTimerP1 = setTimeout(async () => {
    const box = document.getElementById('p1-split-preview');
    if (!box || tipoCorteP1() !== 'partes') return;
    let dur = null, nombre = '';
    if (prevP1.dur) {
      dur = prevP1.dur;
      nombre = document.getElementById('p1-prev-nombre').textContent;
    } else if (colaP1.length) {
      const it = colaP1[0];
      nombre = it.nombre;
      if (!it.url) it.url = URL.createObjectURL(it.file);
      dur = await duracionLocal(it.url, 'f:' + it.id);
    } else {
      const sel = document.getElementById('p1-sel-existente');
      if (sel && sel.value) {
        nombre = sel.value;
        dur = await duracionLocal('/audios/' + encodeURIComponent(sel.value), 'a:' + sel.value);
      }
    }
    if (!dur) { box.textContent = ''; return; }
    const partes = calcularPartesCliente(dur);
    if (!partes.length) { box.textContent = `${nombre} dura ${fmtHMS(dur)}.`; return; }
    const muestra = partes.slice(0, 4).map(([a, b]) => `${fmtHMS(a)}–${fmtHMS(b)}`).join(' · ');
    box.textContent = `${nombre} (${fmtHMS(dur)}) → ${partes.length} ${partes.length === 1 ? 'parte' : 'partes'}: ${muestra}${partes.length > 4 ? ' · …' : ''}`;
  }, 250);
}

function actualizarUiCorteP1() {
  const partes = tipoCorteP1() === 'partes';
  document.getElementById('p1-split-box').style.display = partes ? 'block' : 'none';
  document.getElementById('p1-lbl-tramo').innerHTML = partes
    ? 'Dividir solo un tramo <span style="font-weight:400;color:var(--muted)">(opcional · vacío = archivo completo)</span>'
    : 'Recorte <span style="font-weight:400;color:var(--muted)">(opcional · vacío = archivo completo)</span>';
  const modo = document.getElementById('p1-modo-salida').value;
  document.getElementById('p1-split-exacto-box').style.display = (partes && modo !== 'mp3') ? 'flex' : 'none';
  btnP1Convertir.textContent = partes ? 'Dividir en partes' : 'Procesar archivo(s)';
  document.getElementById('btn-p1-recortar-existente').textContent = partes ? 'Dividir este archivo' : 'Recortar este archivo';
  actualizarPreviewSplitP1();
  renderTimelineP1();
}
document.querySelectorAll('input[name="p1-tipo-corte"], input[name="p1-split-modo"]').forEach(r => r.addEventListener('change', actualizarUiCorteP1));
['p1-split-puntos', 'p1-split-duracion', 'p1-desde', 'p1-hasta'].forEach(id =>
  document.getElementById(id).addEventListener('input', () => { actualizarPreviewSplitP1(); renderTimelineP1(); }));
document.getElementById('p1-split-puntos').addEventListener('focus', () => { document.querySelector('input[name="p1-split-modo"][value="puntos"]').checked = true; actualizarPreviewSplitP1(); });
document.getElementById('p1-split-duracion').addEventListener('focus', () => { document.querySelector('input[name="p1-split-modo"][value="duracion"]').checked = true; actualizarPreviewSplitP1(); });
document.getElementById('p1-modo-salida').addEventListener('change', actualizarUiCorteP1);
document.getElementById('p1-sel-existente').addEventListener('change', actualizarPreviewSplitP1);

function renderPartesP1(nombreOrigen, partes) {
  const todos = [];
  const filas = partes.map(pt => {
    const links = [];
    if (pt.nombre_video) { todos.push(pt.nombre_video); links.push(`<a class="btn-outline btn-xs" href="/audios/${encodeURIComponent(pt.nombre_video)}" download>MP4</a>`); }
    if (pt.nombre_mp3) { todos.push(pt.nombre_mp3); links.push(`<a class="btn-outline btn-xs" href="/audios/${encodeURIComponent(pt.nombre_mp3)}" download>MP3</a>`); }
    const nom = pt.nombre_mp3 || pt.nombre_video;
    return `<div class="parte-item">
      <span class="pn">${String(pt.n).padStart(2, '0')}</span>
      <div><div class="pt">${fmtHMS(pt.inicio)} → ${fmtHMS(pt.fin)} <span style="color:var(--muted)">(${fmtHMS(pt.fin - pt.inicio)})</span></div>
      <div style="font-size:0.76rem;color:var(--muted);overflow-wrap:anywhere">${escHtml(nom)}</div></div>
      <div class="row" style="gap:4px;flex-wrap:nowrap">${links.join('')}</div>
    </div>`;
  }).join('');
  const qs = todos.map(n => 'n=' + encodeURIComponent(n)).join('&');
  return `<div class="row-between">
      <strong style="color:var(--ok)">✔ ${escHtml(nombreOrigen)} dividido en ${partes.length} partes</strong>
      <a class="btn-primary btn-sm" style="text-decoration:none" href="/api/zip_partes?${qs}">Descargar todas (ZIP)</a>
    </div>
    <div style="font-size:0.78rem;color:var(--muted);margin-top:4px">Quedaron guardadas en <code>audios/</code>: las podés transcribir en el paso 2.</div>
    <div class="partes-lista">${filas}</div>`;
}

window.seleccionarMp3EIrAPaso2 = async function(nomMp3) {
  await cargarAudiosDisponibles(nomMp3);
  irAPaso(2);
};

// ==============================================================================
// PASO 2: TRANSCRIBIR CON WHISPER O SUBIR TRANSCRIPCIÓN (TACTIQ / YOUTUBE / SRT)
// ==============================================================================
async function cargarAudiosDisponibles(seleccionarNombre = '') {
  try {
    const r = await fetch('/api/audios_existentes');
    const lista = await r.json();
    const selA = document.getElementById('p2-sel-mp3');
    const selB = document.getElementById('p2-trans-mp3');
    const selP1 = document.getElementById('p1-sel-existente');
    const soloAudios = lista.filter(a => a.tipo !== 'video');

    if (selP1) {
      selP1.innerHTML = lista.length
        ? lista.map(a => `<option value="${escHtml(a.nombre)}">${a.tipo === 'video' ? '🎬 ' : '🎵 '}${escHtml(a.nombre)} (${fmtMB(a.bytes)})</option>`).join('')
        : '<option value="">(Carpeta audios/ vacía)</option>';
    }

    const icono = a => a.tipo === 'video' ? '🎬 ' : '🎵 ';
    const medios = [...soloAudios, ...lista.filter(a => a.tipo === 'video')];
    if (!medios.length) {
      selA.innerHTML = '<option value="">(Todavía no hay audios ni videos en audios/ — cargá uno en Video / Audio)</option>';
    } else {
      selA.innerHTML = medios.map(a => `<option value="${escHtml(a.nombre)}" ${a.nombre === seleccionarNombre ? 'selected' : ''}>${icono(a)}${escHtml(a.nombre)} (${fmtMB(a.bytes)})</option>`).join('');
    }
    selB.innerHTML = '<option value="">Sin audio (solo transcripción y subtítulos)</option>' +
      medios.map(a => `<option value="${escHtml(a.nombre)}">${icono(a)}${escHtml(a.nombre)} (${fmtMB(a.bytes)})</option>`).join('');
  } catch (e) {}
}

async function cargarSesionesGuardadas() {
  try {
    const r = await fetch('/api/sesiones_guardadas');
    const lista = await r.json();
    const cont = document.getElementById('lista-sesiones-guardadas');
    if (!lista.length) {
      cont.innerHTML = '<span style="font-size:0.84rem;color:var(--muted)">Todavía no hay transcripciones guardadas. Creá una con la Opción A o subí una con la Opción B abajo.</span>';
      return;
    }
    cont.innerHTML = lista.map(s => {
      const estado = !s.nombre_mp3 ? '<span class="badge badge-ambar">sin video</span>'
        : s.sincronizado ? '<span class="badge badge-verde">✓ vinculada y sincronizada</span>'
        : '<span class="badge badge-ambar">vinculada · falta sincronizar</span>';
      return `
      <div class="ses-item">
        <div style="min-width:0">
          <strong style="overflow-wrap:anywhere">${escHtml(s.nombre_original)}</strong>
          <div style="font-size:0.78rem;color:var(--muted);margin-top:2px">
            ${fmtHMS(s.duracion)} · ${s.cantidad_segmentos} líneas · ${s.cantidad_hablantes} voces · ${escHtml(s.modelo)}${s.nombre_mp3 ? ` · ${escHtml(s.nombre_mp3)}` : ''}
          </div>
          <div style="margin-top:4px">${estado}</div>
        </div>
        <div class="row" style="gap:6px;flex-wrap:nowrap">
          <button type="button" class="btn-outline btn-xs" onclick="abrirSesionGuardada('${escHtml(s.sesion_id)}', 2)">Vincular / sincronizar</button>
          <button type="button" class="btn-outline btn-xs" onclick="abrirSesionGuardada('${escHtml(s.sesion_id)}', 3)">Hablantes</button>
          <button type="button" class="btn-primary btn-xs" onclick="abrirSesionGuardada('${escHtml(s.sesion_id)}', 4)">Resultados</button>
        </div>
      </div>`;
    }).join('');
  } catch (e) {}
}

window.abrirSesionGuardada = async function(sid, pasoDestino = 3) {
  const r = await fetch('/api/sesion?id=' + encodeURIComponent(sid));
  if (!r.ok) { alert('No se pudo cargar la sesión'); return; }
  const d = await r.json();
  activarSesion(d);
  irAPaso(pasoDestino);
  if (pasoDestino === 2) {
    await renderVinculoP2();
    document.getElementById('p2-vinculo').scrollIntoView({ behavior: 'smooth', block: 'start' });
  }
};

// Opción A: Transcribir con Whisper
// Subir un audio o video para transcribir (queda guardado en audios/)
(function () {
  const zona = document.getElementById('zona-p2-audio');
  const inp = document.getElementById('arch-p2-audio');
  const lbl = document.getElementById('lbl-p2-audio');
  async function subir(file) {
    if (!file) return;
    lbl.textContent = `Subiendo ${file.name}…`;
    zona.style.pointerEvents = 'none';
    try {
      const r = await fetch('/api/subir_medio?nombre=' + encodeURIComponent(file.name), { method: 'POST', body: file });
      if (!r.ok) throw new Error(await r.text());
      const d = await r.json();
      await cargarAudiosDisponibles(d.nombre);
      lbl.textContent = `✔ ${d.nombre} (elegido abajo)`;
    } catch (e) {
      lbl.textContent = 'Subí un audio o video';
      alert('No se pudo subir: ' + e.message);
    } finally {
      zona.style.pointerEvents = '';
    }
  }
  zona.onclick = () => inp.click();
  zona.onkeydown = e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); inp.click(); } };
  zona.ondragover = e => { e.preventDefault(); zona.classList.add('sobre'); };
  zona.ondragleave = () => zona.classList.remove('sobre');
  zona.ondrop = e => { e.preventDefault(); zona.classList.remove('sobre'); subir(e.dataTransfer.files[0]); };
  inp.onchange = () => { subir(inp.files[0]); inp.value = ''; };
})();

document.getElementById('btn-p2-transcribir').onclick = async () => {
  const nomMp3 = document.getElementById('p2-sel-mp3').value;
  if (!nomMp3) { alert('Elegí primero un archivo MP3 o convertí tu video en el Paso 1.'); return; }
  const modelo = document.getElementById('p2-modelo').value;
  const hablantes = document.getElementById('p2-hablantes').value;

  const btn = document.getElementById('btn-p2-transcribir');
  const prog = document.getElementById('p2-progreso-whisper');
  btn.disabled = true;
  prog.innerHTML = `
    <div class="progreso-item">
      <div style="display:flex;justify-content:space-between;font-weight:700;font-size:0.88rem">
        <span class="msg">Iniciando transcripción con Whisper…</span>
        <span class="pct">5%</span>
      </div>
      <div class="barra-fondo"><div class="barra-relleno" style="width:5%"></div></div>
    </div>
  `;

  try {
    const q = new URLSearchParams({
      nombre: nomMp3,
      usar_mp3_existente: '1',
      solo_extraer: '0',
      modelo,
      hablantes,
      desde: document.getElementById('p2-desde').value.trim(),
      hasta: document.getElementById('p2-hasta').value.trim()
    });
    const r = await fetch('/api/iniciar?' + q.toString(), { method: 'POST' });
    if (!r.ok) throw new Error(await r.text());
    const { job_id } = await r.json();

    while (true) {
      await new Promise(res => setTimeout(res, 850));
      const st = await fetch('/api/estado?id=' + encodeURIComponent(job_id));
      const d = await st.json();
      prog.querySelector('.msg').textContent = d.mensaje || 'Procesando…';
      prog.querySelector('.pct').textContent = (d.progreso || 0) + '%';
      prog.querySelector('.barra-relleno').style.width = (d.progreso || 0) + '%';
      if (d.estado === 'error') throw new Error(d.error || 'Error');
      if (d.estado === 'completado') {
        prog.innerHTML = '';
        activarSesion(d.resultado);
        irAPaso(3);
        break;
      }
    }
  } catch (e) {
    prog.innerHTML = `<div style="color:var(--danger);font-weight:700;margin-top:8px">✘ ${escHtml(e.message)}</div>`;
  } finally {
    btn.disabled = false;
  }
};

// Opción B: Subir transcripción existente (Tactiq, YouTube, Whisper, SRT)
let archivoTransSubir = null;
const zonaP2Trans = document.getElementById('zona-p2-trans');
const archP2Trans = document.getElementById('arch-p2-trans');
const btnP2SubirTrans = document.getElementById('btn-p2-subir-trans');

zonaP2Trans.onclick = () => archP2Trans.click();
zonaP2Trans.ondragover = e => { e.preventDefault(); zonaP2Trans.classList.add('sobre'); };
zonaP2Trans.ondragleave = () => zonaP2Trans.classList.remove('sobre');
zonaP2Trans.ondrop = e => {
  e.preventDefault();
  zonaP2Trans.classList.remove('sobre');
  if (e.dataTransfer.files.length) seleccionarArchivoTrans(e.dataTransfer.files[0]);
};
archP2Trans.onchange = () => {
  if (archP2Trans.files.length) seleccionarArchivoTrans(archP2Trans.files[0]);
};

function seleccionarArchivoTrans(f) {
  archivoTransSubir = f;
  document.getElementById('lbl-arch-trans').textContent = `📄 Archivo listo: ${f.name} (${fmtMB(f.size)})`;
  btnP2SubirTrans.disabled = false;
}

btnP2SubirTrans.onclick = async () => {
  if (!archivoTransSubir) return;
  btnP2SubirTrans.disabled = true;
  const nomMp3 = document.getElementById('p2-trans-mp3').value;
  const resDiv = document.getElementById('p2-res-import');
  resDiv.innerHTML = '<div style="margin-top:8px;font-size:0.86rem">⏳ Detectando formato y cargando transcripción…</div>';

  try {
    const q = new URLSearchParams({
      nombre: archivoTransSubir.name,
      nombre_mp3: nomMp3,
      desde: document.getElementById('p2-trans-desde').value.trim(),
      hasta: document.getElementById('p2-trans-hasta').value.trim(),
      reiniciar_reloj: document.getElementById('p2-trans-reset').checked ? '1' : '0'
    });
    const r = await fetch('/api/subir_transcripcion?' + q.toString(), {
      method: 'POST',
      body: archivoTransSubir
    });
    if (!r.ok) throw new Error(await r.text());
    const d = await r.json();
    resDiv.innerHTML = `
      <div style="margin-top:10px;padding:10px;border-radius:8px;background:#f0fdf4;border:1px solid #86efac;font-size:0.86rem">
        ✔ <strong>Formato detectado automáticamente: ${escHtml(d.formato_detectado)}</strong> ·
        ${d.segmentos.length} fragmentos · ${d.hablantes.length} hablantes detectados.
      </div>
    `;
    activarSesion(d);
    await renderVinculoP2();
    document.getElementById('p2-vinculo').scrollIntoView({ behavior: 'smooth', block: 'start' });
  } catch (e) {
    resDiv.innerHTML = `<div style="margin-top:8px;color:var(--danger);font-weight:700">✘ ${escHtml(e.message)}</div>`;
  } finally {
    btnP2SubirTrans.disabled = false;
  }
};

// ==============================================================================
// PASO 3: HABLANTES, GRUPO CON NOMBRE, AUTO-MATCHEO INICIAL Y MI VOZ
// ==============================================================================
function activarSesion(datos) {
  const rd = datos.rango_detectado || {};
  sesionActiva = {
    ...datos,
    miVozId: null,
    muestraIdx: {},
    pararEn: null,
    filtroTexto: '',
    filtroSpk: '',
    rangoInicio: rd.inicio_seg ?? null,
    rangoFin: rd.fin_seg ?? null,
    matchIni: rd.match_ini ?? null,
    matchFin: rd.match_fin ?? null,
    offsetSub: 0
  };
  for (const h of (sesionActiva.hablantes || [])) h.silenciado = false;
  sesionActiva.offset = Number(datos.offset) || 0;
  sesionActiva.sincronizado = Boolean(datos.sincronizado);
  sesionActiva.origen_media = datos.origen_media || '';
  if (typeof pub !== 'undefined') { pub.modo = {}; pub.idxMuestra = {}; }
  sesionActiva.miVozId = detectarMiVoz(sesionActiva.sesion_id, sesionActiva.hablantes || []);
}

// "Soy yo": se recuerda por transcripción y por nombre, para reconocerte en las próximas
function normNombre(t) { return String(t || '').normalize('NFD').replace(/[\u0300-\u036f]/g, '').toLowerCase().trim(); }
function detectarMiVoz(sid, hablantes) {
  try {
    const porSesion = localStorage.getItem('ea_mivoz_' + sid);
    if (porSesion && hablantes.some(h => h.id === porSesion)) return porSesion;
    const miNombre = normNombre(localStorage.getItem('ea_mi_nombre'));
    if (miNombre) {
      const h = hablantes.find(x => normNombre(x.nombre) === miNombre);
      if (h) return h.id;
    }
  } catch (e) {}
  return null;
}
function recordarMiVoz(sid, h) {
  try {
    if (h) {
      localStorage.setItem('ea_mivoz_' + sid, h.id);
      if (h.nombre && !/^hablante\s*\d+$/i.test(h.nombre) && !/^spk_/i.test(h.nombre)) localStorage.setItem('ea_mi_nombre', h.nombre);
    } else {
      localStorage.removeItem('ea_mivoz_' + sid);
    }
  } catch (e) {}
}

async function cargarGruposYListado() {
  try {
    const r = await fetch('/api/voces_registradas');
    const d = await r.json();
    listaGrupos = d.grupos || [];
    if (!listaGrupos.length) {
      listaGrupos = [{ nombre: 'General', cantidad_nombres: 0, cantidad_voces: 0, nombres: [], perfiles: [] }];
    }
    const selG = document.getElementById('p3-sel-grupo');
    if (!listaGrupos.some(g => g.nombre === grupoActivo)) {
      grupoActivo = listaGrupos[0].nombre;
    }
    selG.innerHTML = listaGrupos.map(g =>
      `<option value="${escHtml(g.nombre)}" ${g.nombre === grupoActivo ? 'selected' : ''}>${escHtml(g.nombre)} (${g.cantidad_nombres} nombres · ${g.cantidad_voces} voces)</option>`
    ).join('');

    const gObj = listaGrupos.find(g => g.nombre === grupoActivo) || listaGrupos[0];
    nombresGrupoActual = gObj.nombres || [];
    const dl = document.getElementById('p3-dl-nombres');
    if (dl) dl.innerHTML = nombresGrupoActual.map(n => `<option value="${escHtml(n)}"></option>`).join('');
    document.getElementById('badge-info-grupo').textContent =
      `Grupo "${gObj.nombre}": ${nombresGrupoActual.length} nombres en lista · ${(gObj.perfiles || []).length} voces con huella`;

    const chips = document.getElementById('chips-nombres-grupo');
    if (!nombresGrupoActual.length) {
      chips.innerHTML = '<span style="font-size:0.78rem;color:var(--muted)">Todavía no hay nombres cargados en este grupo.</span>';
    } else {
      chips.innerHTML = nombresGrupoActual.map(n => `
        <span class="badge" style="display:inline-flex;align-items:center;gap:4px">
          ${escHtml(n)}
          <button type="button" style="border:none;background:none;color:#dc2626;cursor:pointer;padding:0;font-size:0.75rem" onclick="borrarNombreDeGrupo('${escHtml(n)}')" title="Quitar nombre">✕</button>
        </span>
      `).join('');
    }
  } catch (e) {}
}

document.getElementById('p3-sel-grupo').onchange = async e => {
  grupoActivo = e.target.value;
  await cargarGruposYListado();
  if (sesionActiva) renderHablantesPaso3();
};

document.getElementById('btn-nuevo-grupo').onclick = async () => {
  const nom = prompt('Nombre del nuevo Grupo de Hablantes (ej. "Equipo Principal", "Grupo Martes", "Proyecto A"):');
  if (!nom || !nom.trim()) return;
  await fetch('/api/crear_grupo', {
    method: 'POST',
    headers: {'Content-Type':'application/json'},
    body: JSON.stringify({ nombre: nom.trim() })
  });
  grupoActivo = nom.trim();
  await cargarGruposYListado();
  if (sesionActiva) renderHablantesPaso3();
};

window.borrarNombreDeGrupo = async function(nom) {
  await fetch('/api/eliminar_nombre_listado', {
    method: 'POST',
    headers: {'Content-Type':'application/json'},
    body: JSON.stringify({ nombre: nom, grupo: grupoActivo })
  });
  await cargarGruposYListado();
  if (sesionActiva) renderHablantesPaso3();
};

// Subir Excel / CSV / TXT de nombres de hablantes
document.getElementById('arch-listado-nombres').onchange = async e => {
  const f = e.target.files[0];
  if (!f) return;
  e.target.value = '';
  const q = new URLSearchParams({ nombre: f.name, grupo: grupoActivo });
  const r = await fetch('/api/subir_listado_nombres?' + q.toString(), { method: 'POST', body: f });
  if (!r.ok) { alert(await r.text()); return; }
  const d = await r.json();
  alert(`¡Listo! Se cargaron ${d.agregados} nombres de hablantes en el grupo "${grupoActivo}".`);
  await cargarGruposYListado();
  if (sesionActiva) renderHablantesPaso3();
};

document.getElementById('btn-pegar-nombres').onclick = async () => {
  const txt = prompt('Pegá los nombres de los hablantes separados por coma o uno debajo del otro:');
  if (!txt || !txt.trim()) return;
  const blob = new Blob([txt.replace(/,/g, '\n')], { type: 'text/plain' });
  const q = new URLSearchParams({ nombre: 'pegado.txt', grupo: grupoActivo });
  const r = await fetch('/api/subir_listado_nombres?' + q.toString(), { method: 'POST', body: blob });
  if (r.ok) {
    await cargarGruposYListado();
    if (sesionActiva) renderHablantesPaso3();
  }
};

// Auto-matchear por presentación en el audio ("mi nombre es...", "soy...")
document.getElementById('btn-auto-presentacion').onclick = async () => {
  if (!sesionActiva) return;
  const resBox = document.getElementById('res-auto-presentacion');
  resBox.innerHTML = '⏳ Buscando presentaciones en la transcripción…';
  const r = await fetch('/api/auto_matchear_presentaciones', {
    method: 'POST',
    headers: {'Content-Type':'application/json'},
    body: JSON.stringify({ sesion_id: sesionActiva.sesion_id, grupo: grupoActivo })
  });
  if (!r.ok) { resBox.innerHTML = '✘ Error'; return; }
  const d = await r.json();
  const antes = Object.fromEntries(sesionActiva.hablantes.map(h => [h.id, h.nombre]));
  sesionActiva.segmentos = d.segmentos;
  sesionActiva.hablantes = d.hablantes;
  sesionActiva.ultimosCambios = cambiosDeNombres(antes, d.hablantes);
  if (!d.hallazgos.length) {
    resBox.innerHTML = `<span style="color:#b45309">No se encontraron frases tipo "soy…" o "mi nombre es…". Probá con Tactiq o escribí los nombres en la lista.</span>`;
  } else {
    resBox.innerHTML = `<span style="color:var(--ok);font-weight:700">✔ Se detectaron ${d.hallazgos.length} presentaciones automáticas: ${d.hallazgos.map(h => escHtml(h.nombre_detectado)).join(', ')}.</span>`;
  }
  await cargarGruposYListado();
  renderPaso3();
};

// Auto-sincronizar y matchear con Tactiq
document.getElementById('arch-tactiq-sync').onchange = async e => {
  const f = e.target.files[0];
  if (!f || !sesionActiva) return;
  e.target.value = '';
  const resBox = document.getElementById('res-tactiq-sync');
  resBox.innerHTML = '⏳ Sincronizando tiempos de Tactiq con el audio y asignando hablantes…';
  const q = new URLSearchParams({
    sesion_id: sesionActiva.sesion_id,
    nombre: f.name,
    grupo: grupoActivo
  });
  const r = await fetch('/api/sincronizar_tactiq_voces?' + q.toString(), { method: 'POST', body: f });
  if (!r.ok) {
    resBox.innerHTML = `<span style="color:var(--danger)">✘ ${escHtml(await r.text())}</span>`;
    return;
  }
  const d = await r.json();
  const antes = Object.fromEntries(sesionActiva.hablantes.map(h => [h.id, h.nombre]));
  sesionActiva.segmentos = d.segmentos;
  sesionActiva.hablantes = d.hablantes;
  sesionActiva.ultimosCambios = cambiosDeNombres(antes, d.hablantes);
  if (!sesionActiva.miVozId) sesionActiva.miVozId = detectarMiVoz(sesionActiva.sesion_id, d.hablantes);
  const absOff = Math.round(Math.abs(Number(d.offset_segundos || 0)) * 10) / 10;
  resBox.innerHTML = `
    <div style="color:var(--ok);font-weight:700;margin-bottom:4px">
      ✔ Sincronizado (${Object.keys(d.mapeo_hablantes || {}).length} hablantes etiquetados) · Diferencia detectada entre archivos: ${fmtHMS(absOff)} (${absOff}s).
    </div>
    <div style="display:flex;gap:6px;flex-wrap:wrap;align-items:center;font-size:0.78rem;color:#1e3a8a">
      <span>¿Querés llevar esta diferencia al Paso 4 (Subtítulos del Video)?</span>
      <button type="button" class="btn-outline btn-xs" onclick="aplicarDesfaseDesdePaso3(${absOff})">🎬 El Video empezó ${fmtHMS(absOff)} ANTES (+${absOff}s)</button>
      <button type="button" class="btn-outline btn-xs" onclick="aplicarDesfaseDesdePaso3(${-absOff})">📄 La Transcripción empezó ${fmtHMS(absOff)} ANTES (-${absOff}s)</button>
    </div>
  `;
  await cargarGruposYListado();
  renderPaso3();
};

function cambiosDeNombres(antes, hablantes) {
  return hablantes.filter(h => antes[h.id] !== undefined && antes[h.id] !== h.nombre)
    .map(h => ({ id: h.id, de: antes[h.id], a: h.nombre }));
}

function renderPaso3() {
  if (!sesionActiva) {
    document.getElementById('p3-vacio').style.display = 'block';
    document.getElementById('p3-contenido').style.display = 'none';
    return;
  }
  document.getElementById('p3-vacio').style.display = 'none';
  document.getElementById('p3-contenido').style.display = 'block';

  const audioEl = document.getElementById('p3-audio');
  const pBar = document.getElementById('p3-player-bar');
  if (sesionActiva.nombre_mp3) {
    pBar.style.display = 'flex';
    const srcEsperado = '/audios/' + encodeURIComponent(sesionActiva.nombre_mp3);
    if (!audioEl.getAttribute('src') || !audioEl.getAttribute('src').endsWith(encodeURIComponent(sesionActiva.nombre_mp3))) {
      audioEl.src = srcEsperado;
    }
  } else {
    pBar.style.display = 'none';
  }

  audioEl.ontimeupdate = () => {
    if (sesionActiva && sesionActiva.pararEn !== null && audioEl.currentTime >= sesionActiva.pararEn) {
      audioEl.pause();
      sesionActiva.pararEn = null;
      document.getElementById('p3-player-info').textContent = 'Muestra finalizada.';
    }
  };

  audioEl.onerror = () => {
    if (audioEl.getAttribute('src')) document.getElementById('p3-player-info').textContent = `No se pudo abrir "${sesionActiva?.nombre_mp3}". ¿Sigue en la carpeta audios/? Revisalo en Transcripción.`;
  };
  cargarGruposYListado().then(() => {
    renderHablantesPaso3();
    renderSegmentosPaso3();
    actualizarResumenAudioPaso3();
  });
}

function recalcularEstadisticasSesion() {
  if (!sesionActiva) return;
  for (const h of sesionActiva.hablantes) {
    const segs = sesionActiva.segmentos.filter(s => s.hablante === h.id);
    h.cantidad = segs.length;
    h.duracion = Math.round(segs.reduce((acc, s) => acc + Math.max(0, s.fin - s.inicio), 0) * 10) / 10;
    const ordenados = [...segs].sort((a, b) => (b.fin - b.inicio) - (a.fin - a.inicio));
    h.muestras = ordenados.slice(0, 5).map(m => ({ inicio: m.inicio, fin: m.fin, texto: m.texto.slice(0, 95) }));
  }
}

function renderDiagnosticoClaridad() {
  const st = sesionActiva;
  document.getElementById('p3-ctx-nombre').textContent = st.nombre_original || st.sesion_id;
  const yo = st.hablantes.find(h => h.id === st.miVozId);
  const sinNombre = st.hablantes.filter(h => /^(hablante\s*\d+|spk_\d+)$/i.test(String(h.nombre || '').trim())).length;
  const partes = [
    `${st.hablantes.length} ${st.hablantes.length === 1 ? 'voz' : 'voces'}`,
    fmtHMS(st.duracion),
    st.nombre_mp3 ? `${esVideoNombre(st.nombre_mp3) ? 'video' : 'audio'}: ${escHtml(st.nombre_mp3)}${st.sincronizado ? '' : ' <span class="pend">(sin sincronizar)</span>'}` : '<span class="pend">sin video ni audio</span>',
    yo ? `<span class="ok">tu voz: ${escHtml(yo.nombre)}</span>` : '<span class="pend">tu voz: sin marcar</span>',
    sinNombre ? `<span class="pend">${sinNombre} sin nombre</span>` : '<span class="ok">todas con nombre</span>'
  ];
  document.getElementById('p3-ctx-datos').innerHTML = partes.join(' · ');
  document.getElementById('p3-aviso-media').innerHTML = !st.nombre_mp3
    ? 'Esta transcripción no tiene video ni audio vinculado, por eso no se puede escuchar. <a href="#" onclick="irAPaso(2);return false">Vincularlo en Transcripción →</a>'
    : (!st.sincronizado ? 'La transcripción todavía no está sincronizada con su video: lo que escuches puede no coincidir con el texto. <a href="#" onclick="irAPaso(2);return false">Sincronizar →</a>' : '');
  // El panel de "completar automáticamente" se abre solo si faltan nombres
  const auto = document.getElementById('p3-auto-box');
  if (auto && !auto.dataset.tocado) auto.open = sinNombre > 0;
}

function renderHablantesPaso3() {
  const st = sesionActiva;
  recalcularEstadisticasSesion();
  renderDiagnosticoClaridad();

  const grid = document.getElementById('p3-hablantes-grid');
  const selFiltro = document.getElementById('p3-filtro-spk');
  const filtroActual = selFiltro.value;
  selFiltro.innerHTML = '<option value="">Todas las voces</option>';
  grid.innerHTML = '';

  const cambios = Object.fromEntries((st.ultimosCambios || []).map(c => [c.id, c]));
  const ordenadas = [...st.hablantes].sort((a, b) => (b.duracion || 0) - (a.duracion || 0));
  for (const h of st.hablantes) {
    const opt = document.createElement('option');
    opt.value = h.id;
    opt.textContent = `${h.nombre} (${h.cantidad})`;
    selFiltro.appendChild(opt);
  }

  for (const h of ordenadas) {
    const esYo = st.miVozId === h.id;
    const row = document.createElement('div');
    row.className = 'voz-row' + (esYo ? ' es-yo' : '');
    const ejemplo = (h.muestras && h.muestras[0]) ? `“${escHtml(h.muestras[0].texto)}”` : '';
    const cambio = cambios[h.id];
    row.innerHTML = `
      <span class="dot" style="background:${h.color}"></span>
      <div class="voz-nombre">
        <input type="text" class="inp-nom-manual" list="p3-dl-nombres" value="${escHtml(h.nombre)}" aria-label="Nombre de esta voz">
        <div class="voz-ej">${ejemplo}</div>
      </div>
      <div class="voz-stats">
        ${h.cantidad} ${h.cantidad === 1 ? 'intervención' : 'intervenciones'} · ${fmtHMS(h.duracion)}
        ${cambio ? `<br><span class="cambio">✨ antes "${escHtml(cambio.de)}"</span>` : (h.auto_etiquetado ? '<br><span class="guardada">✓ guardada en el grupo</span>' : '')}
      </div>
      <button type="button" class="btn-outline btn-xs btn-muestra" ${(!st.nombre_mp3 || !h.muestras.length) ? 'disabled title="Esta transcripción no tiene audio vinculado"' : ''}>▶ Escuchar</button>
      <button type="button" class="btn-yo btn-mi-voz ${esYo ? 'activo' : ''}" aria-pressed="${esYo}">${esYo ? '★ Soy yo' : '☆ Soy yo'}</button>
    `;

    const inp = row.querySelector('.inp-nom-manual');
    const guardar = async () => {
      const nuevo = inp.value.trim();
      if (!nuevo || nuevo === h.nombre) { inp.value = h.nombre; return; }
      h.nombre = nuevo;
      inp.disabled = true;
      try {
        const r = await fetch('/api/guardar_voz_registro', {
          method: 'POST', headers: {'Content-Type':'application/json'},
          body: JSON.stringify({ sesion_id: st.sesion_id, spk_id: h.id, nombre: nuevo, color: h.color, grupo: grupoActivo })
        });
        if (!r.ok) throw new Error(await r.text());
        h.auto_etiquetado = true;
        if (st.miVozId === h.id) recordarMiVoz(st.sesion_id, h);
        if (st.ultimosCambios) st.ultimosCambios = st.ultimosCambios.filter(c => c.id !== h.id);
        await cargarGruposYListado();
      } catch (e) {
        alert('No se pudo guardar el nombre: ' + e.message);
      }
      renderHablantesPaso3(); renderSegmentosPaso3(); actualizarResumenAudioPaso3();
    };
    inp.addEventListener('change', guardar);
    inp.addEventListener('keydown', e => { if (e.key === 'Enter') inp.blur(); });

    const btnMuestra = row.querySelector('.btn-muestra');
    btnMuestra.onclick = () => {
      if (!h.muestras.length) return;
      const idx = (st.muestraIdx[h.id] || 0) % h.muestras.length;
      st.muestraIdx[h.id] = idx + 1;
      const m = h.muestras[idx];
      btnMuestra.textContent = `▶ ${idx + 1}/${h.muestras.length}`;
      reproducirFragmentoPaso3(m.inicio, m.fin, `${h.nombre}: "${m.texto}"`);
    };

    row.querySelector('.btn-mi-voz').onclick = () => {
      st.miVozId = esYo ? null : h.id;
      recordarMiVoz(st.sesion_id, esYo ? null : h);
      renderHablantesPaso3(); actualizarResumenAudioPaso3();
    };
    grid.appendChild(row);
  }
  selFiltro.value = filtroActual;
  renderVocesP3b();
}

document.getElementById('p3-auto-box').addEventListener('toggle', e => { e.currentTarget.dataset.tocado = '1'; });

// Botones rápidos para separar en 2 a 30 hablantes
document.getElementById('p3-k-otro').onchange = e => { document.getElementById('btn-p3-k-otro').dataset.k = e.target.value; };
document.querySelectorAll('.btn-recluster-k').forEach(btn => {
  btn.onclick = async () => {
    if (!sesionActiva) return;
    const k = Number(btn.dataset.k);
    btn.disabled = true;
    try {
      const r = await fetch('/api/reagrupar', {
        method: 'POST',
        headers: {'Content-Type':'application/json'},
        body: JSON.stringify({ sesion_id: sesionActiva.sesion_id, num_hablantes: k })
      });
      if (!r.ok) throw new Error(await r.text());
      const d = await r.json();
      sesionActiva.hablantes = d.hablantes;
      sesionActiva.segmentos = d.segmentos;
      sesionActiva.miVozId = detectarMiVoz('__nada__', d.hablantes);
      sesionActiva.ultimosCambios = [];
      renderPaso3();
    } catch (e) {
      alert('No se pudo reagrupar: ' + e.message);
    } finally {
      btn.disabled = false;
    }
  };
});

document.getElementById('btn-p3-silenciar-todos').onclick = () => {
  if (!sesionActiva) return;
  for (const h of sesionActiva.hablantes) h.silenciado = true;
  renderHablantesPaso3(); renderSegmentosPaso3(); actualizarResumenAudioPaso3();
};
document.getElementById('btn-p3-activar-todos').onclick = () => {
  if (!sesionActiva) return;
  for (const h of sesionActiva.hablantes) h.silenciado = false;
  renderHablantesPaso3(); renderSegmentosPaso3(); actualizarResumenAudioPaso3();
};
document.getElementById('btn-p3-nuevo-spk').onclick = () => {
  if (!sesionActiva) return;
  const num = sesionActiva.hablantes.length + 1;
  const colores = ["#2563eb", "#059669", "#d97706", "#7c3aed", "#db2777", "#0891b2"];
  sesionActiva.hablantes.push({
    id: 'SPK_' + Date.now().toString().slice(-4),
    nombre: `Hablante ${num}`,
    color: colores[(num - 1) % colores.length],
    silenciado: false, cantidad: 0, duracion: 0, muestras: []
  });
  renderHablantesPaso3(); renderSegmentosPaso3();
};

function reproducirFragmentoPaso3(ini, fin, etiqueta) {
  const audioEl = document.getElementById('p3-audio');
  const info = document.getElementById('p3-player-info');
  if (!sesionActiva || !sesionActiva.nombre_mp3) {
    info.textContent = 'Esta transcripción no tiene video ni audio vinculado: vinculalo en Transcripción.';
    return;
  }
  const off = offSes(sesionActiva);
  const arrancar = () => {
    sesionActiva.pararEn = fin + off;
    audioEl.currentTime = Math.max(0, ini + off);
    const pr = audioEl.play();
    if (pr && pr.catch) pr.catch(err => { info.textContent = 'No se pudo reproducir: ' + err.message; });
    info.textContent = `▶ ${etiqueta} (${fmtHMS(ini + off)} → ${fmtHMS(fin + off)} del video)`;
  };
  if (audioEl.readyState >= 1) arrancar();
  else {
    info.textContent = 'Cargando el audio…';
    audioEl.addEventListener('loadedmetadata', arrancar, { once: true });
    audioEl.load();
  }
}

function esSegSilenciado(st, seg) {
  if (seg.override_mute !== undefined && seg.override_mute !== null) return Boolean(seg.override_mute);
  const h = st.hablantes.find(x => x.id === seg.hablante);
  return Boolean(h && h.silenciado);
}

function offSes(st) { return Number((st || sesionActiva || {}).offset) || 0; }
// El tramo se expresa en tiempo del VIDEO; las líneas, en tiempo de la transcripción (+ desfase)
function estaEnRango(st, seg) {
  const off = offSes(st);
  const r0 = st.rangoInicio ?? -Infinity;
  const r1 = st.rangoFin ?? Infinity;
  return seg.fin + off > r0 && seg.inicio + off < r1;
}

function renderSegmentosPaso3() {
  const st = sesionActiva;
  if (!st) return;
  const wrap = document.getElementById('p3-seg-wrap');
  wrap.innerHTML = '';

  const mapaH = {};
  for (const h of st.hablantes) mapaH[h.id] = h;

  const ocultarFuera = document.getElementById('p3-chk-ocultar-fuera') && document.getElementById('p3-chk-ocultar-fuera').checked;
  const filtrados = st.segmentos.filter(s => {
    if (ocultarFuera && !estaEnRango(st, s)) return false;
    if (st.filtroSpk && s.hablante !== st.filtroSpk) return false;
    if (st.filtroTexto && !s.texto.toLowerCase().includes(st.filtroTexto)) return false;
    return true;
  });

  document.getElementById('p3-cont-seg').textContent = filtrados.length;

  for (const seg of filtrados) {
    const sil = esSegSilenciado(st, seg);
    const enR = estaEnRango(st, seg);
    const spk = mapaH[seg.hablante] || { nombre: seg.hablante, color: '#64748b' };

    const row = document.createElement('div');
    let cls = 'seg-row';
    if (sil) cls += ' seg-silenciado';
    if (!enR) cls += ' fuera-de-rango';
    if (st.rangoInicio !== null && Math.abs(seg.inicio - st.rangoInicio) < 1.5) cls += ' marca-inicio';
    if (st.rangoFin !== null && Math.abs(seg.fin - st.rangoFin) < 1.5) cls += ' marca-fin';
    row.className = cls;

    const opts = st.hablantes.map(h =>
      `<option value="${h.id}" ${h.id === seg.hablante ? 'selected' : ''}>${escHtml(h.nombre)}</option>`
    ).join('');

    row.innerHTML = `
      <button type="button" class="seg-time-btn" ${st.nombre_mp3 ? '' : 'disabled title="Sin video o audio vinculado"'}>▶ ${fmtHMS(seg.inicio + offSes(st))}</button>
      <select style="border-left:4px solid ${spk.color};font-weight:700;font-size:0.82rem">${opts}</select>
      <div class="seg-texto" contenteditable="true" spellcheck="false" style="font-size:0.9rem;padding:4px">${escHtml(seg.texto)}</div>
      <div style="display:flex;gap:4px">
        <button type="button" class="btn-outline btn-xs btn-ini" title="El tramo empieza en esta línea">⛳</button>
        <button type="button" class="btn-outline btn-xs btn-fin" title="El tramo termina en esta línea">🏁</button>
        <button type="button" class="btn-outline btn-xs btn-mute" title="${sil ? 'Línea excluida: tocá para volver a incluirla' : 'Excluir esta línea del audio y la transcripción que generes'}">${sil ? '🚫' : '✓'}</button>
      </div>
    `;

    row.querySelector('.seg-time-btn').onclick = () => reproducirFragmentoPaso3(seg.inicio, seg.fin, spk.nombre);
    const sel = row.querySelector('select');
    sel.onchange = () => {
      seg.hablante = sel.value;
      delete seg.override_mute;
      renderHablantesPaso3(); renderSegmentosPaso3(); actualizarResumenAudioPaso3();
      programarGuardadoSesion();
    };
    const txtEl = row.querySelector('.seg-texto');
    txtEl.onblur = () => {
      const nuevo = txtEl.textContent.trim();
      if (nuevo !== seg.texto) { seg.texto = nuevo; programarGuardadoSesion(); }
    };
    row.querySelector('.btn-ini').onclick = () => {
      st.rangoInicio = seg.inicio + offSes(st);
      document.getElementById('p3-t-ini').value = fmtCorte(st.rangoInicio);
      renderSegmentosPaso3(); actualizarResumenAudioPaso3();
    };
    row.querySelector('.btn-fin').onclick = () => {
      st.rangoFin = seg.fin + offSes(st);
      document.getElementById('p3-t-fin').value = fmtCorte(Math.ceil(st.rangoFin));
      renderSegmentosPaso3(); actualizarResumenAudioPaso3();
    };
    row.querySelector('.btn-mute').onclick = () => {
      seg.override_mute = !sil;
      renderSegmentosPaso3(); actualizarResumenAudioPaso3();
      programarGuardadoSesion();
    };

    wrap.appendChild(row);
  }
}

document.getElementById('p3-buscar').oninput = e => {
  if (!sesionActiva) return;
  sesionActiva.filtroTexto = e.target.value.toLowerCase();
  renderSegmentosPaso3();
};
document.getElementById('p3-filtro-spk').onchange = e => {
  if (!sesionActiva) return;
  sesionActiva.filtroSpk = e.target.value;
  renderSegmentosPaso3();
};

// Recorte por frases en Paso 3
document.getElementById('btn-p3-buscar-frases').onclick = async () => {
  if (!sesionActiva) return;
  const fIni = document.getElementById('p3-f-ini').value.trim();
  const fFin = document.getElementById('p3-f-fin').value.trim();
  if (!fIni && !fFin) { alert('Escribí una frase de inicio o de fin.'); return; }
  const r = await fetch('/api/buscar_frases', {
    method: 'POST', headers: {'Content-Type':'application/json'},
    body: JSON.stringify({ sesion_id: sesionActiva.sesion_id, frase_ini: fIni, frase_fin: fFin, segmentos: sesionActiva.segmentos })
  });
  const d = await r.json();
  if (d.match_ini) {
    sesionActiva.rangoInicio = d.match_ini.frase_inicio;
    document.getElementById('p3-t-ini').value = fmtHMS(d.match_ini.frase_inicio);
    document.getElementById('p3-pill-ini').innerHTML = `<div style="font-size:0.78rem;color:var(--ok);margin-top:4px">✔ En ${fmtHMS(d.match_ini.frase_inicio)}: "${escHtml(d.match_ini.texto.slice(0,50))}"</div>`;
  }
  if (d.match_fin) {
    sesionActiva.rangoFin = d.match_fin.frase_fin;
    document.getElementById('p3-t-fin').value = fmtHMS(d.match_fin.frase_fin);
    document.getElementById('p3-pill-fin').innerHTML = `<div style="font-size:0.78rem;color:var(--ok);margin-top:4px">✔ En ${fmtHMS(d.match_fin.frase_fin)}: "${escHtml(d.match_fin.texto.slice(0,50))}"</div>`;
  }
  renderSegmentosPaso3(); actualizarResumenAudioPaso3();
};

function aplicarInputsRangoPaso3() {
  if (!sesionActiva) return;
  const vIni = document.getElementById('p3-t-ini').value.trim();
  const vFin = document.getElementById('p3-t-fin').value.trim();
  sesionActiva.rangoInicio = parseTiempoCliente(vIni);
  sesionActiva.rangoFin = parseTiempoCliente(vFin);
  // Sincronizar también con los inputs de recorte de video del Paso 4
  const p4Ini = document.getElementById('p4-t-ini');
  const p4Fin = document.getElementById('p4-t-fin');
  if (p4Ini && !p4Ini.value) p4Ini.value = vIni;
  if (p4Fin && !p4Fin.value) p4Fin.value = vFin;
  renderSegmentosPaso3();
  actualizarResumenAudioPaso3();
}

document.getElementById('p3-t-ini').onchange = aplicarInputsRangoPaso3;
document.getElementById('p3-t-fin').onchange = aplicarInputsRangoPaso3;
document.getElementById('btn-p3-aplicar-rango').onclick = aplicarInputsRangoPaso3;
document.getElementById('p3-chk-ocultar-fuera').onchange = () => renderSegmentosPaso3();

document.getElementById('btn-p3-tomar-ini-audio').onclick = () => {
  const a = document.getElementById('pub-audio');
  if (!a) return;
  document.getElementById('p3-t-ini').value = fmtCorte(a.currentTime || 0);
  aplicarInputsRangoPaso3();
};
document.getElementById('btn-p3-tomar-fin-audio').onclick = () => {
  const a = document.getElementById('pub-audio');
  if (!a) return;
  document.getElementById('p3-t-fin').value = fmtCorte(Math.ceil(a.currentTime || 0));
  aplicarInputsRangoPaso3();
};

document.getElementById('btn-p3-limpiar-rango').onclick = () => {
  if (!sesionActiva) return;
  sesionActiva.rangoInicio = null; sesionActiva.rangoFin = null;
  document.getElementById('p3-t-ini').value = ''; document.getElementById('p3-t-fin').value = '';
  document.getElementById('p3-pill-ini').innerHTML = ''; document.getElementById('p3-pill-fin').innerHTML = '';
  renderSegmentosPaso3(); actualizarResumenAudioPaso3();
};

window.descargarTranscripcionRecortadaPaso3 = function(formato) {
  if (!sesionActiva) return;
  aplicarInputsRangoPaso3();
  const st = sesionActiva;
  const off = offSes(st);
  const conRango = st.rangoInicio !== null || st.rangoFin !== null;
  const [r0, r1] = rangoP3();
  const resetReloj = document.getElementById('p3-chk-reset-reloj').checked;
  const shift = (resetReloj && conRango && r0 > 0) ? r0 : 0;
  const mapaNom = {};
  for (const h of st.hablantes) mapaNom[h.id] = h.nombre;
  const segs = st.segmentos.filter(s => estaEnRango(st, s) && !esSegSilenciado(st, s));
  if (!segs.length) { alert('No hay líneas dentro de ese tramo.'); return; }
  let contenido = '', ext;
  if (formato === 'srt') {
    const { bloques } = construirBloquesSubtitulosSincronizados(st, off - shift, true, false, false, { maxChars: 84, minDur: 1.2, maxDur: 6.0 });
    contenido = bloques.map((b, i) => `${i + 1}\n${fmtSrt(b.ini, ',')} --> ${fmtSrt(b.fin, ',')}\n${b.textoSrt || b.textoPlano}\n`).join('\n');
    ext = '_transcripcion.srt';
  } else if (formato === 'txt_limpio') {
    contenido = segs.map(s => `${mapaNom[s.hablante] || s.hablante}: ${s.texto}`).join('\n');
    ext = '_transcripcion_texto.txt';
  } else {
    contenido = segs.map(s => {
      const ini = Math.max(0, Math.max(s.inicio + off, conRango ? r0 : -Infinity) - shift);
      const fin = Math.max(ini + 0.2, Math.min(s.fin + off, conRango ? r1 : Infinity) - shift);
      return `[${fmtHMS(ini)} - ${fmtHMS(fin)}] ${mapaNom[s.hablante] || s.hablante}: ${s.texto}`;
    }).join('\n');
    ext = '_transcripcion_tiempos.txt';
  }
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([contenido], { type: 'text/plain;charset=utf-8' }));
  a.download = `${st.nombre_original.replace(/\.[^.]+$/, '')}${conRango ? '_tramo' : ''}${ext}`;
  a.click();
};

document.getElementById('btn-p3-recortar-sesion-perm').onclick = async () => {
  if (!sesionActiva) return;
  const desde = document.getElementById('p3-t-ini').value.trim();
  const hasta = document.getElementById('p3-t-fin').value.trim();
  if (!desde && !hasta) {
    alert('Primero escribí el minuto y segundo en "Desde" y/o "Hasta" para recortar la transcripción.');
    return;
  }
  const reset = document.getElementById('p3-chk-reset-reloj').checked;
  if (!confirm(`¿Dejar en la transcripción de esta sesión únicamente el tramo desde ${desde || '00:00'} hasta ${hasta || 'el final'}?`)) return;

  await guardarSesionServidor();
  const off = offSes(sesionActiva);
  const tD = desde ? Math.max(0, parseTiempoCliente(desde) - off) : null;
  const tH = hasta ? Math.max(0, parseTiempoCliente(hasta) - off) : null;
  const r = await fetch('/api/recortar_transcripcion_sesion', {
    method: 'POST',
    headers: {'Content-Type':'application/json'},
    body: JSON.stringify({
      sesion_id: sesionActiva.sesion_id,
      desde: tD !== null ? tD.toFixed(2) : '',
      hasta: tH !== null ? tH.toFixed(2) : '',
      reiniciar_reloj: reset
    })
  });
  if (!r.ok) { alert(await r.text()); return; }
  const d = await r.json();
  sesionActiva.segmentos = d.segmentos;
  sesionActiva.hablantes = d.hablantes;
  sesionActiva.duracion = d.duracion;
  if (reset) {
    // La transcripción ahora empieza en 00:00: hay que volver a sincronizarla con su video
    await guardarVinculoP2({ offset: tD !== null ? Math.round((tD + off) * 10) / 10 : off, sincronizado: !!sesionActiva.nombre_mp3 });
  }
  sesionActiva.rangoInicio = null;
  sesionActiva.rangoFin = null;
  document.getElementById('p3-t-ini').value = '';
  document.getElementById('p3-t-fin').value = '';
  renderPaso3();
  renderPaso4();
};

function durMediaP3() {
  const a = document.getElementById('pub-audio');
  const st = sesionActiva;
  return (a && isFinite(a.duration) && a.duration > 0) ? a.duration : (st.duracion || 0) + Math.max(0, offSes(st));
}
function rangoP3() {
  const st = sesionActiva;
  return [st.rangoInicio ?? 0, st.rangoFin ?? durMediaP3()];
}
function durEnRango(segs) {
  const [r0, r1] = rangoP3();
  const off = offSes(sesionActiva);
  return segs.reduce((acc, s) => acc + Math.max(0, Math.min(s.fin + off, r1) - Math.max(s.inicio + off, r0)), 0);
}
function lineaExcluida(s) { return s.override_mute === true; }

function actualizarResumenAudioPaso3() {
  if (!sesionActiva) return;
  const st = sesionActiva;
  const [r0, r1] = rangoP3();
  const badgeRango = document.getElementById('p3-badge-rango');
  if (st.rangoInicio !== null || st.rangoFin !== null) {
    badgeRango.textContent = `Tramo: ${fmtHMS(r0)} → ${fmtHMS(r1)} (${fmtHMS(r1 - r0)})`;
    badgeRango.className = 'badge badge-verde';
  } else {
    badgeRango.textContent = `Tramo: todo (${fmtHMS(durMediaP3())})`;
    badgeRango.className = 'badge badge-azul';
  }
  renderTimelineP3();

  // A · Mi voz limpia
  const aviso = document.getElementById('p3-mivoz-aviso');
  const btnA = document.getElementById('btn-p3-export-audio');
  const yo = st.hablantes.find(h => h.id === st.miVozId);
  const minSeg = Number(document.getElementById('p3-min-seg').value || 0);
  const maxMin = Number(document.getElementById('p3-max-min').value || 0);
  if (!st.nombre_mp3) {
    aviso.innerHTML = 'Esta transcripción no tiene audio vinculado.';
    btnA.disabled = true;
    document.getElementById('p3-resumen-audio').innerHTML = '';
  } else if (!yo) {
    aviso.innerHTML = 'Primero marcá <strong>☆ Soy yo</strong> en tu voz (paso 1).';
    btnA.disabled = true;
    document.getElementById('p3-resumen-audio').innerHTML = '';
  } else {
    aviso.innerHTML = '';
    btnA.disabled = false;
    let mios = st.segmentos.filter(s => s.hablante === yo.id && !lineaExcluida(s) && estaEnRango(st, s));
    if (minSeg > 0) mios = mios.filter(s => (Math.min(s.fin + offSes(st), r1) - Math.max(s.inicio + offSes(st), r0)) >= minSeg);
    let dur = durEnRango(mios);
    if (maxMin > 0) dur = Math.min(dur, maxMin * 60);
    document.getElementById('p3-resumen-audio').innerHTML = `${mios.length} fragmentos de <strong>${escHtml(yo.nombre)}</strong> · sale ≈ ${fmtHMS(dur)}`;
  }

  // B · Audio sin ciertas voces
  const q = vocesElegidasP3b();
  const btnB = document.getElementById('btn-p3b-export');
  btnB.disabled = !st.nombre_mp3;
  const quitar = st.segmentos.filter(s => estaEnRango(st, s) && (q.has(s.hablante) || lineaExcluida(s)));
  const resB = document.getElementById('p3b-resumen');
  if (!st.nombre_mp3) resB.innerHTML = 'Esta transcripción no tiene audio vinculado.';
  else if (!quitar.length) resB.innerHTML = `Nadie marcado: se recorta el audio al tramo (${fmtHMS(r1 - r0)}).`;
  else {
    const modo = document.getElementById('p3b-modo').value;
    resB.innerHTML = `${modo === 'silenciar' ? 'Se silencian' : 'Se quitan'} ${quitar.length} fragmentos (${fmtHMS(durEnRango(quitar))})` +
      (modo === 'recortar' ? ` · sale ≈ ${fmtHMS(Math.max(0, (r1 - r0) - durEnRango(quitar)))}` : '');
  }
}

['p3-fmt-audio', 'p3-max-min', 'p3-min-seg', 'p3b-modo', 'p3b-fmt'].forEach(id => {
  document.getElementById(id).onchange = () => actualizarResumenAudioPaso3();
});

// Línea de tiempo del tramo en el paso 3
function renderTimelineP3() {
  const st = sesionActiva;
  const capa = document.getElementById('p3-tl-marcas');
  const dur = st ? durMediaP3() : 0;
  if (!st || !capa || !dur) return;
  const pct = t => `${Math.max(0, Math.min(100, t / dur * 100))}%`;
  const [r0, r1] = rangoP3();
  capa.innerHTML = (st.rangoInicio !== null || st.rangoFin !== null)
    ? `<div class="tl-rango" style="left:${pct(r0)};width:calc(${pct(r1)} - ${pct(r0)})"></div>` : '';
  actualizarCabezalP3();
}
function actualizarCabezalP3() {
  const st = sesionActiva;
  const a = document.getElementById('pub-audio');
  const cab = document.getElementById('p3-tl-cabezal');
  if (!st || !a || !cab) return;
  const dur = durMediaP3();
  if (dur) cab.style.left = `calc(${Math.min(100, (a.currentTime || 0) / dur * 100)}% - 1px)`;
}
document.getElementById('pub-audio').addEventListener('timeupdate', actualizarCabezalP3);
document.getElementById('pub-audio').addEventListener('loadedmetadata', () => { if (sesionActiva) actualizarResumenAudioPaso3(); });
document.getElementById('p3-timeline').addEventListener('click', e => {
  const st = sesionActiva;
  const a = document.getElementById('pub-audio');
  if (!st || !st.nombre_mp3) return;
  const dur = durMediaP3();
  const r = e.currentTarget.getBoundingClientRect();
  a.currentTime = Math.max(0, Math.min(dur, (e.clientX - r.left) / r.width * dur));
});

// B · elegir voces
function vocesElegidasP3b() {
  const st = sesionActiva;
  if (!st) return new Set();
  if (!st.quitarVoces) st.quitarVoces = new Set();
  return st.quitarVoces;
}
function renderVocesP3b() {
  const st = sesionActiva;
  const cont = document.getElementById('p3b-voces');
  if (!st || !cont) return;
  const q = vocesElegidasP3b();
  cont.innerHTML = [...st.hablantes].sort((a, b) => (b.duracion || 0) - (a.duracion || 0)).map(h => `
    <label class="check"><input type="checkbox" data-spk="${escHtml(h.id)}" ${q.has(h.id) ? 'checked' : ''}>
      <span><span class="dot-mini" style="background:${h.color}"></span>${escHtml(h.nombre)}${h.id === st.miVozId ? ' <em>(vos)</em>' : ''}</span></label>`).join('');
  cont.querySelectorAll('input').forEach(c => c.onchange = () => {
    c.checked ? q.add(c.dataset.spk) : q.delete(c.dataset.spk);
    actualizarResumenAudioPaso3();
  });
}
document.getElementById('btn-p3b-todos-menos-yo').onclick = () => {
  const st = sesionActiva;
  if (!st) return;
  if (!st.miVozId) { alert('Primero marcá "☆ Soy yo" en tu voz (en Hablantes).'); return; }
  st.quitarVoces = new Set(st.hablantes.filter(h => h.id !== st.miVozId).map(h => h.id));
  renderVocesP3b(); actualizarResumenAudioPaso3();
};
document.getElementById('btn-p3b-solo-yo').onclick = () => {
  const st = sesionActiva;
  if (!st) return;
  if (!st.miVozId) { alert('Primero marcá "☆ Soy yo" en tu voz (en Hablantes).'); return; }
  st.quitarVoces = new Set([st.miVozId]);
  renderVocesP3b(); actualizarResumenAudioPaso3();
};
document.getElementById('btn-p3b-ninguno').onclick = () => {
  if (!sesionActiva) return;
  sesionActiva.quitarVoces = new Set();
  renderVocesP3b(); actualizarResumenAudioPaso3();
};

function mostrarAudioGenerado(box, d) {
  box.innerHTML = `
    <div class="salida-ok">
      <div style="font-weight:600;color:var(--ok);margin-bottom:6px">✔ ${escHtml(d.nombre_archivo)} (${fmtHMS(d.duracion_final)})</div>
      ${/\.zip$/i.test(d.nombre_archivo) ? '' : `<audio controls src="/audios/${encodeURIComponent(d.nombre_archivo)}" style="width:100%;height:34px;margin-bottom:6px"></audio>`}
      <a class="btn-primary btn-sm" href="/audios/${encodeURIComponent(d.nombre_archivo)}" download>Descargar</a>
    </div>`;
}
async function exportarAudioP3(payload, btn, box) {
  btn.disabled = true;
  const txt = btn.textContent;
  btn.textContent = 'Generando…';
  box.innerHTML = '';
  try {
    const r = await fetch('/api/exportar_mp3', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify(payload) });
    if (!r.ok) throw new Error(await r.text());
    mostrarAudioGenerado(box, await r.json());
  } catch (e) {
    box.innerHTML = `<div style="margin-top:8px;color:var(--danger);font-weight:600">✘ ${escHtml(e.message)}</div>`;
  } finally {
    btn.disabled = false;
    btn.textContent = txt;
  }
}

// A · Mi voz limpia
document.getElementById('btn-p3-export-audio').onclick = () => {
  const st = sesionActiva;
  if (!st || !st.nombre_mp3 || !st.miVozId) return;
  const off = offSes(st);
  const mios = st.segmentos.filter(s => s.hablante === st.miVozId && !lineaExcluida(s));
  exportarAudioP3({
    sesion_id: st.sesion_id, modo: 'aislar',
    formato: document.getElementById('p3-fmt-audio').value,
    max_minutos: Number(document.getElementById('p3-max-min').value || 0),
    min_dur_seg: Number(document.getElementById('p3-min-seg').value || 0),
    pausa_seg: 0.3, rango_inicio: st.rangoInicio, rango_fin: st.rangoFin,
    intervalos: [],
    activos: mios.map(s => ({ inicio: Math.max(0, s.inicio + off), fin: s.fin + off, texto: s.texto, confianza: s.confianza || 0.8 })).filter(s => s.fin > 0)
  }, document.getElementById('btn-p3-export-audio'), document.getElementById('p3-res-audio'));
};

// B · Audio sin ciertas voces
document.getElementById('btn-p3b-export').onclick = () => {
  const st = sesionActiva;
  if (!st || !st.nombre_mp3) return;
  const q = vocesElegidasP3b();
  const off = offSes(st);
  const quitar = st.segmentos.filter(s => q.has(s.hablante) || lineaExcluida(s));
  const modo = quitar.length ? document.getElementById('p3b-modo').value : 'rango_directo';
  if (modo === 'rango_directo' && st.rangoInicio === null && st.rangoFin === null) {
    alert('Marcá al menos una voz para sacar o elegí un tramo.');
    return;
  }
  exportarAudioP3({
    sesion_id: st.sesion_id, modo,
    formato: document.getElementById('p3b-fmt').value,
    max_minutos: 0, min_dur_seg: 0, pausa_seg: 0.3,
    rango_inicio: st.rangoInicio, rango_fin: st.rangoFin,
    intervalos: quitar.map(s => [Math.max(0, s.inicio + off), s.fin + off]).filter(x => x[1] > 0),
    activos: st.segmentos.filter(s => !quitar.includes(s)).map(s => ({ inicio: Math.max(0, s.inicio + off), fin: s.fin + off, texto: s.texto, confianza: s.confianza || 0.8 })).filter(s => s.fin > 0)
  }, document.getElementById('btn-p3b-export'), document.getElementById('p3b-res'));
};


// ==============================================================================
// PASO 2 · VINCULAR LA TRANSCRIPCIÓN CON SU VIDEO / AUDIO Y SINCRONIZAR
// ==============================================================================
const EXT_VIDEO_GEN = /\.(mp4|mkv|mov|avi|webm)$/i;
function esVideoNombre(n) { return EXT_VIDEO_GEN.test(n || ''); }

async function renderVinculoP2() {
  const card = document.getElementById('p2-vinculo');
  if (!sesionActiva) { card.style.display = 'none'; return; }
  card.style.display = 'block';
  const st = sesionActiva;
  try {
    const lista = await (await fetch('/api/sesiones_guardadas')).json();
    const sel = document.getElementById('p4-sel-sesion-activa');
    sel.innerHTML = lista.map(x => `<option value="${escHtml(x.sesion_id)}" ${x.sesion_id === st.sesion_id ? 'selected' : ''}>${escHtml(x.nombre_original)} · ${fmtHMS(x.duracion)} · ${x.cantidad_hablantes} voces${x.nombre_mp3 ? (x.sincronizado ? ' · ✓ sincronizada' : ' · vinculada') : ' · sin video'}</option>`).join('');
  } catch (e) {}
  let medios = [];
  try { medios = await (await fetch('/api/audios_existentes')).json(); } catch (e) {}
  const selM = document.getElementById('p2v-media');
  const actual = st.nombre_mp3 || '';
  const ops = medios.map(a => `<option value="${escHtml(a.nombre)}" ${a.nombre === actual ? 'selected' : ''}>${a.tipo === 'video' ? '🎬 ' : '🎵 '}${escHtml(a.nombre)}</option>`);
  if (actual && !medios.some(a => a.nombre === actual)) ops.unshift(`<option value="${escHtml(actual)}" selected>⚠ ${escHtml(actual)} (no está en audios/)</option>`);
  selM.innerHTML = '<option value="">— Sin video ni audio —</option>' + ops.join('');
  const selRef = document.getElementById('p4-sel-frase-ref');
  const mapaN = Object.fromEntries(st.hablantes.map(h => [h.id, h.nombre]));
  selRef.innerHTML = st.segmentos.filter(x => (x.texto || '').trim().length >= 8).map(x =>
    `<option value="${x.inicio}">[${fmtHMS(x.inicio)}] ${escHtml(mapaN[x.hablante] || '')}: ${escHtml(x.texto.slice(0, 70))}</option>`).join('');
  document.getElementById('p4-offset-seg').value = st.offset || 0;
  cargarMediaP2();
  actualizarEstadoVinculoP2();
}

function cargarMediaP2() {
  const st = sesionActiva;
  const v = document.getElementById('p4-video');
  const nom = st && st.nombre_mp3;
  document.getElementById('p2v-sin-media').style.display = nom ? 'none' : 'block';
  v.style.display = nom ? 'block' : 'none';
  if (!nom) { if (v.getAttribute('src')) { v.removeAttribute('src'); v.load(); } return; }
  const src = '/audios/' + encodeURIComponent(nom);
  if (v.getAttribute('src') !== src) v.src = src;
  v.style.height = esVideoNombre(nom) ? '' : '54px';
}

function actualizarEstadoVinculoP2() {
  const st = sesionActiva;
  if (!st) return;
  const badge = document.getElementById('p2v-estado');
  const info = document.getElementById('p2v-media-info');
  const sync = document.getElementById('p2v-sync');
  const ex = document.getElementById('p2v-explica');
  if (!st.nombre_mp3) {
    badge.className = 'badge badge-ambar';
    badge.textContent = 'Sin video ni audio';
    info.textContent = 'Sin video o audio vinculado solo vas a poder sacar textos y subtítulos (no escuchar las voces ni silenciarlas).';
    sync.style.display = 'none';
    return;
  }
  sync.style.display = 'block';
  const tipo = esVideoNombre(st.nombre_mp3) ? 'video' : 'audio';
  info.textContent = st.origen_media === 'whisper'
    ? `La transcripción salió de este ${tipo} con Whisper: ya está sincronizada.`
    : `${tipo === 'video' ? 'Video' : 'Audio'} vinculado. Si la transcripción viene de otro lado (por ej. Tactiq), sincronizala.`;
  if (st.sincronizado) {
    badge.className = 'badge badge-verde';
    badge.textContent = `✓ Sincronizada${st.offset ? ` · desfase ${st.offset > 0 ? '+' : ''}${st.offset} s` : ''}`;
  } else {
    badge.className = 'badge badge-ambar';
    badge.textContent = 'Falta sincronizar';
  }
  const off = Number(st.offset) || 0;
  ex.innerHTML = off === 0
    ? 'Sin desfase: el 00:00 de la transcripción es el 00:00 del video.'
    : off > 0
      ? `La transcripción va <strong>${fmtHMS(off)}</strong> atrasada: lo que marca en 00:00 pasa en el ${fmtHMS(off)} del video.`
      : `La transcripción va <strong>${fmtHMS(-off)}</strong> adelantada: las frases anteriores al inicio del video quedan afuera.`;
  document.getElementById('btn-p2v-ok').textContent = st.sincronizado ? '✓ Sincronizada' : '✓ Está sincronizada (confirmar)';
  document.getElementById('btn-p2v-ok').className = st.sincronizado ? 'btn-outline btn-sm btn-block' : 'btn-secondary btn-sm btn-block';
}

async function guardarVinculoP2(campos) {
  const st = sesionActiva;
  if (!st) return;
  try {
    const r = await fetch('/api/sesion_vincular', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ sesion_id: st.sesion_id, ...campos })
    });
    if (!r.ok) throw new Error(await r.text());
    const d = await r.json();
    st.nombre_mp3 = d.nombre_mp3; st.offset = d.offset; st.sincronizado = d.sincronizado; st.origen_media = d.origen_media;
    document.getElementById('p2v-guardado').textContent = 'Guardado ✓';
  } catch (e) {
    alert('No se pudo guardar: ' + e.message);
  }
  actualizarEstadoVinculoP2();
}
let _tOffsetP2 = null;
function cambiarOffsetP2(valor, guardarYa = false) {
  const st = sesionActiva;
  if (!st) return;
  valor = Math.round(Number(valor || 0) * 10) / 10;
  st.offset = valor;
  document.getElementById('p4-offset-seg').value = valor;
  actualizarEstadoVinculoP2();
  actualizarSubVivoP2();
  clearTimeout(_tOffsetP2);
  _tOffsetP2 = setTimeout(() => guardarVinculoP2({ offset: valor, sincronizado: true }), guardarYa ? 0 : 700);
}

document.getElementById('p4-sel-sesion-activa').onchange = e => abrirSesionGuardada(e.target.value, 2);
document.getElementById('p2v-media').onchange = async e => {
  await guardarVinculoP2({ nombre_media: e.target.value, offset: 0, sincronizado: false });
  document.getElementById('p4-offset-seg').value = 0;
  cargarMediaP2();
};
document.getElementById('p2v-arch').onchange = async e => {
  const f = e.target.files[0];
  e.target.value = '';
  if (!f) return;
  const info = document.getElementById('p2v-media-info');
  info.textContent = `Subiendo ${f.name}…`;
  try {
    const r = await fetch('/api/subir_medio?nombre=' + encodeURIComponent(f.name), { method: 'POST', body: f });
    if (!r.ok) throw new Error(await r.text());
    const d = await r.json();
    await guardarVinculoP2({ nombre_media: d.nombre, offset: 0, sincronizado: false });
    await renderVinculoP2();
  } catch (err) {
    info.textContent = '';
    alert('No se pudo subir: ' + err.message);
  }
};

const videoP4 = document.getElementById('p4-video');
function actualizarSubVivoP2() {
  const st = sesionActiva;
  const t = videoP4.currentTime || 0;
  document.getElementById('p2v-t').textContent = fmtCorte(t);
  const subEl = document.getElementById('p4-sub-live');
  if (!st) { subEl.style.display = 'none'; return; }
  const off = Number(st.offset) || 0;
  const seg = st.segmentos.find(x => x.inicio + off <= t && t < x.fin + off);
  if (seg) {
    const h = st.hablantes.find(x => x.id === seg.hablante);
    subEl.textContent = `${h ? '[' + h.nombre + '] ' : ''}${seg.texto}`;
    subEl.style.display = 'inline-block';
  } else {
    subEl.style.display = 'none';
  }
}
videoP4.addEventListener('timeupdate', actualizarSubVivoP2);
videoP4.addEventListener('seeked', actualizarSubVivoP2);
videoP4.addEventListener('loadedmetadata', () => {
  document.getElementById('p2v-dur').textContent = isFinite(videoP4.duration) ? fmtCorte(videoP4.duration) : '--:--';
});
document.querySelectorAll('[data-salto-v]').forEach(b => b.onclick = () => {
  videoP4.currentTime = Math.max(0, (videoP4.currentTime || 0) + Number(b.dataset.saltoV));
});
document.getElementById('btn-p2v-ir-frase').onclick = () => {
  if (!sesionActiva) return;
  const t = Number(document.getElementById('p4-sel-frase-ref').value || 0);
  videoP4.currentTime = Math.max(0, t + (Number(sesionActiva.offset) || 0));
  videoP4.play().catch(() => {});
};
document.getElementById('btn-p4-tomar-ancla-video').onclick = () => {
  document.getElementById('p4-t-video-ancla').value = fmtCorte(videoP4.currentTime || 0);
};
document.getElementById('btn-p4-sync-video').onclick = () => {
  if (!sesionActiva) return;
  const tFrase = Number(document.getElementById('p4-sel-frase-ref').value || 0);
  const txt = document.getElementById('p4-t-video-ancla').value.trim();
  let tVideo = txt ? parseTiempoCliente(txt) : (videoP4.currentTime || 0);
  if (tVideo === null || isNaN(tVideo)) { alert('Tiempo inválido. Ejemplo: 02:35'); return; }
  if (!txt) document.getElementById('p4-t-video-ancla').value = fmtCorte(tVideo);
  cambiarOffsetP2(tVideo - tFrase, true);
};
document.getElementById('btn-p2v-ok').onclick = () => guardarVinculoP2({ offset: Number(sesionActiva?.offset) || 0, sincronizado: true });
document.getElementById('p4-offset-seg').oninput = e => cambiarOffsetP2(e.target.value);
window.ajustarOffsetSub = function(delta, reset = false) {
  if (!sesionActiva) return;
  cambiarOffsetP2(reset ? 0 : (Number(sesionActiva.offset) || 0) + delta);
};
window.invertirSignoOffsetSub = function() {
  if (!sesionActiva) return;
  cambiarOffsetP2(-(Number(sesionActiva.offset) || 0));
};
window.aplicarDesfaseDesdePaso3 = function(offsetSeg) {
  if (!sesionActiva) return;
  irAPaso(2);
  cambiarOffsetP2(offsetSeg, true);
};

// ==============================================================================
// GUARDAR CORRECCIONES DE LA TRANSCRIPCIÓN (hablante por línea, texto, exclusiones)
// ==============================================================================
let _tGuardarSes = null;
function programarGuardadoSesion() {
  clearTimeout(_tGuardarSes);
  _tGuardarSes = setTimeout(guardarSesionServidor, 1200);
}
async function guardarSesionServidor() {
  clearTimeout(_tGuardarSes);
  const st = sesionActiva;
  if (!st) return;
  try {
    await fetch('/api/sesion_guardar', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        sesion_id: st.sesion_id,
        segmentos: st.segmentos.map(x => ({ hablante: x.hablante, texto: x.texto, override_mute: x.override_mute === true })),
        hablantes: st.hablantes.map(h => ({ id: h.id, nombre: h.nombre, color: h.color }))
      })
    });
  } catch (e) {}
}

// ==============================================================================
// PASO 4 · RESULTADOS
// ==============================================================================
async function renderPaso4() {
  if (!sesionActiva) {
    document.getElementById('p4-vacio').style.display = 'block';
    document.getElementById('p4-contenido').style.display = 'none';
    return;
  }
  document.getElementById('p4-vacio').style.display = 'none';
  document.getElementById('p4-contenido').style.display = 'block';
  const st = sesionActiva;
  recalcularEstadisticasSesion();
  const yo = st.hablantes.find(h => h.id === st.miVozId);
  const tipo = st.nombre_mp3 ? (esVideoNombre(st.nombre_mp3) ? 'video' : 'audio') : '';
  const sinNombre = st.hablantes.filter(h => /^(hablante\s*\d+|spk_\d+)$/i.test(String(h.nombre || '').trim())).length;
  document.getElementById('res-material').innerHTML = `
    <div class="fila"><span class="k">Transcripción</span><strong>${escHtml(st.nombre_original || st.sesion_id)}</strong></div>
    <div class="fila"><span class="k">${tipo === 'audio' ? 'Audio' : 'Video'}</span>${st.nombre_mp3 ? escHtml(st.nombre_mp3) : '<span class="pend">sin vincular</span> <a href="#" onclick="irAPaso(2);return false">vincular</a>'}</div>
    <div class="fila"><span class="k">Sincronización</span>${!st.nombre_mp3 ? '—' : st.sincronizado ? `<span class="ok">✓ sincronizada${st.offset ? ` (desfase ${st.offset > 0 ? '+' : ''}${st.offset} s)` : ''}</span>` : '<span class="pend">sin confirmar</span> <a href="#" onclick="irAPaso(2);return false">sincronizar</a>'}</div>
    <div class="fila"><span class="k">Voces</span>${st.hablantes.length} · ${yo ? `<span class="ok">vos: ${escHtml(yo.nombre)}</span>` : '<span class="pend">tu voz sin marcar</span>'}${sinNombre ? ` · <span class="pend">${sinNombre} sin nombre</span>` : ''} <a href="#" onclick="irAPaso(3);return false">editar</a></div>`;

  cargarPreviewPub(st.nombre_mp3);
  document.getElementById('pub-preview').style.display = st.nombre_mp3 ? 'block' : 'none';
  pub.sesion = st;
  pub.video = st.nombre_mp3;
  pub.nombres = {};
  pub.miVoz = st.miVozId || detectarMiVoz(st.sesion_id, st.hablantes) || null;
  st.miVozId = pub.miVoz;

  const aviso = document.getElementById('res-video-aviso');
  const esVid = tipo === 'video';
  aviso.innerHTML = !st.nombre_mp3
    ? 'Para generar videos, vinculá el video de esta clase en <a href="#" onclick="irAPaso(2);return false">Transcripción</a>.'
    : !esVid ? 'Esta transcripción está vinculada a un <strong>audio</strong>. Para generar video, vinculá el video en <a href="#" onclick="irAPaso(2);return false">Transcripción</a>.'
    : !st.sincronizado ? 'Para generar el video, la transcripción tiene que estar <strong>sincronizada</strong> con este video (si no, se silenciarían o subtitularían momentos equivocados). <button type="button" class="btn-primary btn-xs" onclick="abrirSesionGuardada(sesionActiva.sesion_id, 2)">Sincronizar ahora</button> Podés ir eligiendo las voces y probar en la vista previa.' : '';
  const listoVideo = esVid && st.sincronizado;
  document.getElementById('btn-p4-unir-video').disabled = !listoVideo;
  document.getElementById('res-video-card').classList.toggle('deshabilitada', !listoVideo);
  document.getElementById('pub-resumen-video').textContent = '';

  renderVocesPub();
  renderVocesP3b();
  actualizarResumenAudioPaso3();
}

function fmtSrt(seg, sepMs = ',') {
  const msTotal = Math.max(0, Math.round(seg * 1000));
  const h = Math.floor(msTotal / 3600000);
  const m = Math.floor((msTotal % 3600000) / 60000);
  const s = Math.floor((msTotal % 60000) / 1000);
  const ms = msTotal % 1000;
  return `${String(h).padStart(2,'0')}:${String(m).padStart(2,'0')}:${String(s).padStart(2,'0')}${sepMs}${String(ms).padStart(3,'0')}`;
}

function dividirTextoEnFrasesSubJS(texto, maxChars = 84) {
  let limpio = String(texto || '').replace(/https?:\/\/\S{28,}/g, m => m.slice(0, 24) + '…');
  limpio = limpio.replace(/\s+/g, ' ').trim();
  if (!limpio) return [];
  if (limpio.length <= maxChars) return [limpio];

  const oraciones = limpio.split(/(?<=[.?!;:])\s+/);
  const fragmentos = [];
  let actual = '';

  const agregarPalabras = (bloque) => {
    for (let pal of bloque.split(' ')) {
      if (!pal) continue;
      if (pal.length > maxChars) pal = pal.slice(0, maxChars - 1) + '…';
      if (!actual) {
        actual = pal;
      } else if (actual.length + 1 + pal.length <= maxChars) {
        actual += ' ' + pal;
      } else {
        fragmentos.push(actual);
        actual = pal;
      }
    }
  };

  for (const orac of oraciones) {
    const p = orac.trim();
    if (!p) continue;
    if (!actual && p.length <= maxChars) {
      actual = p;
    } else if (actual && actual.length + 1 + p.length <= maxChars) {
      actual += ' ' + p;
    } else {
      if (actual) { fragmentos.push(actual); actual = ''; }
      if (p.length <= maxChars) {
        actual = p;
      } else {
        for (const cl of p.split(/(?<=,)\s+/)) {
          const c = cl.trim();
          if (!c) continue;
          if (!actual && c.length <= maxChars) {
            actual = c;
          } else if (actual && actual.length + 1 + c.length <= maxChars) {
            actual += ' ' + c;
          } else {
            if (actual) { fragmentos.push(actual); actual = ''; }
            if (c.length <= maxChars) actual = c;
            else agregarPalabras(c);
          }
        }
      }
    }
  }
  if (actual) fragmentos.push(actual);
  return fragmentos;
}

function balancearDosLineasSubJS(texto, maxPorLinea = 42) {
  const t = String(texto || '').replace(/\s+/g, ' ').trim();
  if (t.length <= maxPorLinea || !t.includes(' ')) return t;
  const mitad = Math.floor(t.length / 2);
  let mejorPos = -1;
  let mejorDist = 1e9;
  for (let i = 0; i < t.length; i++) {
    if (t[i] === ' ') {
      const d = Math.abs(i - mitad);
      if (d < mejorDist) {
        mejorDist = d;
        mejorPos = i;
      }
    }
  }
  if (mejorPos <= 0) return t;
  return t.slice(0, mejorPos).trim() + '\n' + t.slice(mejorPos + 1).trim();
}

function construirBloquesSubtitulosSincronizados(st, offset, incNom, incSil, agregarBloque00, opts = {}) {
  const GAP = 0.04; // 40ms de separación mínima para garantizar 0 superposiciones
  const VEL_CHARS = 15.0; // ~15 caracteres por segundo de lectura cómoda
  const maxCharsEl = document.getElementById('p4-max-chars');
  const maxDurEl = document.getElementById('p4-max-dur');
  const maxCharsSub = opts.maxChars || (maxCharsEl ? Number(maxCharsEl.value || 84) : 84);
  const maxPorLinea = Math.max(25, Math.floor(maxCharsSub / 2));
  const minDur = opts.minDur || 1.2;
  const maxDur = opts.maxDur || (maxDurEl ? Number(maxDurEl.value || 6.0) : 6.0);

  const mapaNom = {};
  for (const h of (st.hablantes || [])) mapaNom[h.id] = h.nombre;

  const candidatos = (st.segmentos || []).filter(s => estaEnRango(st, s) && (incSil || !esSegSilenciado(st, s)));
  let descartadosAntesDeCero = 0;

  const crudos = [];
  for (const s of candidatos) {
    const iniReal = s.inicio + offset;
    const finReal = s.fin + offset;
    if (finReal <= 0.05) {
      descartadosAntesDeCero++;
      continue;
    }
    const ini = Math.max(0, iniReal);
    const fin = Math.max(ini + 0.4, finReal);
    crudos.push({
      ini,
      fin,
      iniOrig: s.inicio,
      finOrig: s.fin,
      hablante: mapaNom[s.hablante] || s.hablante,
      texto: String(s.texto || '').trim()
    });
  }

  crudos.sort((a, b) => (a.ini - b.ini) || (a.fin - b.fin));

  const desplazados = [];
  let cursorGlobal = 0;
  const nC = crudos.length;

  for (let idxC = 0; idxC < nC; idxC++) {
    const c = crudos[idxC];
    const pref = (incNom && c.hablante) ? `[${c.hablante}] ` : '';
    const presupuestoTxt = Math.max(20, maxCharsSub - pref.length);
    const partes = dividirTextoEnFrasesSubJS(c.texto, presupuestoTxt);
    if (!partes.length) continue;

    const mPartes = partes.length;
    const iniSeg = Math.max(c.ini, cursorGlobal);

    const dursIdeales = partes.map(pt => {
      const dId = (pref.length + pt.length) / VEL_CHARS;
      return Math.max(minDur, Math.min(maxDur, dId));
    });
    const sumaIdeal = dursIdeales.reduce((a, b) => a + b, 0);

    let huecoNeto;
    if (idxC + 1 < nC) {
      const proxIni = crudos[idxC + 1].ini;
      huecoNeto = (proxIni - iniSeg) - mPartes * GAP;
    } else {
      huecoNeto = Math.max(sumaIdeal, c.fin - iniSeg);
    }

    const durOrigNeta = Math.max(sumaIdeal, Math.min(c.fin - iniSeg, mPartes * maxDur));
    let durTotalUsar;
    if (huecoNeto >= mPartes * minDur) {
      durTotalUsar = Math.min(durOrigNeta, huecoNeto);
    } else {
      durTotalUsar = Math.max(mPartes * minDur, Math.min(sumaIdeal, mPartes * maxDur));
    }

    let cursorSub = iniSeg;
    partes.forEach((pt, idxP) => {
      const prop = dursIdeales[idxP] / Math.max(0.1, sumaIdeal);
      const durP = Math.max(minDur, Math.min(maxDur, durTotalUsar * prop));
      const iniB = Math.round(cursorSub * 1000) / 1000;
      const finB = Math.round((iniB + durP) * 1000) / 1000;
      const textoPlano = pref + pt;
      desplazados.push({
        ini: iniB,
        fin: finB,
        iniOrig: c.iniOrig,
        hablante: c.hablante,
        esBloqueCero: false,
        texto: pt,
        textoPlano,
        textoSrt: balancearDosLineasSubJS(textoPlano, maxPorLinea)
      });
      cursorSub = Math.round((finB + GAP) * 1000) / 1000;
    });

    cursorGlobal = cursorSub;
  }

  const bloques = [];
  if (agregarBloque00 && desplazados.length > 0 && desplazados[0].ini >= 0.5) {
    const tPrimerSub = desplazados[0].ini;
    const msg00 = `[Inicio del video 00:00 — La transcripción comienza en ${fmtHMS(tPrimerSub)}]`;
    bloques.push({
      ini: 0,
      fin: Math.max(0.4, Math.round((tPrimerSub - GAP) * 1000) / 1000),
      iniOrig: null,
      hablante: '',
      esBloqueCero: true,
      texto: msg00,
      textoPlano: msg00,
      textoSrt: balancearDosLineasSubJS(msg00, maxPorLinea)
    });
  }

  for (const d of desplazados) bloques.push(d);

  return {
    bloques,
    descartadosAntesDeCero,
    primerInicioVideo: desplazados.length ? desplazados[0].ini : 0,
    maxCharsSub,
    maxDur
  };
}

function actualizarPreviewSincroPaso4() {}

window.descargarSubtitulosPaso4 = function(formato) {
  if (!sesionActiva) return;
  const st = sesionActiva;
  const offset = Number(st.offset) || 0;
  const incNom = document.getElementById('p4-inc-nom').checked;
  const agregarBloque00 = document.getElementById('p4-chk-bloque-00').checked;
  const { bloques } = construirBloquesSubtitulosSincronizados(st, offset, incNom, false, agregarBloque00);
  if (!bloques.length) { alert('No hay subtítulos en ese tramo.'); return; }
  let contenido = '', ext = '.srt';
  if (formato === 'srt') {
    contenido = bloques.map((b, i) => `${i + 1}\n${fmtSrt(b.ini, ',')} --> ${fmtSrt(b.fin, ',')}\n${b.textoSrt || b.textoPlano}\n`).join('\n');
  } else if (formato === 'vtt') {
    contenido = 'WEBVTT\n\n' + bloques.map(b => `${fmtSrt(b.ini, '.')} --> ${fmtSrt(b.fin, '.')}\n${b.textoSrt || b.textoPlano}\n`).join('\n');
    ext = '.vtt';
  } else {
    contenido = bloques.map(b => `[${fmtHMS(b.ini)} - ${fmtHMS(b.fin)}] ${b.textoPlano}`).join('\n');
    ext = '_tiempos.txt';
  }
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([contenido], { type: 'text/plain;charset=utf-8' }));
  a.download = (st.nombre_mp3 || st.nombre_original).replace(/\.[^.]+$/, '') + ext;
  a.click();
};

document.getElementById('btn-p4-unir-video').onclick = async () => {
  const st = sesionActiva;
  if (!st || !st.nombre_mp3 || !esVideoNombre(st.nombre_mp3)) return;
  if (!st.sincronizado) { alert('Primero sincronizá la transcripción con su video (paso 2).'); return; }
  const btn = document.getElementById('btn-p4-unir-video');
  const resDiv = document.getElementById('p4-res-unir');
  btn.disabled = true;
  resDiv.innerHTML = '⏳ Procesando el video…';
  try {
    await guardarSesionServidor();
    const q = new URLSearchParams({
      sesion_id: st.sesion_id,
      nombre_video: st.nombre_mp3,
      modo: document.getElementById('p4-modo-unir').value,
      offset: String(Number(st.offset) || 0),
      inc_nom: document.getElementById('p4-inc-nom').checked ? '1' : '0',
      inc_sil: '0',
      bloque_00: document.getElementById('p4-chk-bloque-00').checked ? '1' : '0',
      max_chars: document.getElementById('p4-max-chars').value,
      max_dur: document.getElementById('p4-max-dur').value,
      desde: document.getElementById('p3-t-ini').value.trim(),
      hasta: document.getElementById('p3-t-fin').value.trim()
    });
    const r = await fetch('/api/unir_subtitulos_video?' + q.toString(), { method: 'POST' });
    if (!r.ok) throw new Error(await r.text());
    const d = await r.json();
    resDiv.innerHTML = `<div class="salida-ok"><strong style="color:var(--ok)">✔ ${escHtml(d.nombre_archivo)}</strong>
      <div style="margin-top:6px"><a class="btn-primary btn-sm" href="/audios/${encodeURIComponent(d.nombre_archivo)}" download>Descargar</a></div></div>`;
  } catch (e) {
    resDiv.innerHTML = `<div style="color:var(--danger);font-weight:600">✘ ${escHtml(e.message)}</div>`;
  } finally {
    btn.disabled = false;
  }
};

// ==============================================================================
// PESTAÑA 5: ENTRENAR MODELO ML DE MI ESTILO Y CONVERTIR GUIONES
// ==============================================================================
let ultimoPromptAdn = '';

async function cargarEstadoEstiloPaso5() {
  try {
    const [rEst, rSes] = await Promise.all([
      fetch('/api/estilo_estado'),
      fetch('/api/sesiones_guardadas')
    ]);
    const est = await rEst.json();
    const sesiones = await rSes.json();

    const contSes = document.getElementById('p5-lista-sesiones-train');
    if (!sesiones.length) {
      contSes.innerHTML = '<span style="font-size:0.82rem;color:var(--muted)">No hay sesiones guardadas aún. Podés sumar archivos .TXT/.SRT con «+ Archivos extra».</span>';
    } else {
      contSes.innerHTML = sesiones.map((s, idx) => `
        <label style="display:flex;align-items:center;justify-content:space-between;gap:8px;padding:6px 8px;background:#fff;border:1px solid var(--border);border-radius:6px;font-size:0.83rem;cursor:pointer">
          <span style="display:flex;align-items:center;gap:6px">
            <input type="checkbox" class="chk-ses-train" value="${escHtml(s.sesion_id)}" ${idx === 0 ? 'checked' : ''}>
            <strong>${escHtml(s.nombre_original)}</strong>
            <small style="color:var(--muted)">(${s.cantidad_segmentos} frases · ${fmtHMS(s.duracion)})</small>
          </span>
          <select class="sel-spk-train" data-sid="${escHtml(s.sesion_id)}" style="width:auto;padding:3px 6px;font-size:0.78rem">
            <option value="SPK_1" selected>Usar Hablante 1 (Principal / Mi Voz)</option>
            <option value="ACTIVOS">Usar todos los hablantes no silenciados</option>
            <option value="SPK_2">Usar Hablante 2</option>
          </select>
        </label>
      `).join('');
    }

    renderModeloEstiloUI(est);
  } catch (e) {}
}

function renderModeloEstiloUI(est) {
  const badge = document.getElementById('p5-badge-estado');
  if (!est || !est.entrenado) {
    badge.textContent = '⚠️ Modelo aún sin entrenar (hacé clic en Entrenar Modelo)';
    badge.className = 'badge badge-ambar';
    return;
  }
  const m = est.metricas || {};
  badge.textContent = `✔ Modelo ML Entrenado (${m.total_intervenciones || 0} frases · ${m.total_palabras || 0} palabras)`;
  badge.className = 'badge badge-verde';

  document.getElementById('p5-m-frases').textContent = m.total_intervenciones || 0;
  document.getElementById('p5-m-pal').textContent = m.total_palabras || 0;
  document.getElementById('p5-m-long').textContent = (m.palabras_por_oracion_media || 0) + ' pal/oración';

  const chipsM = document.getElementById('p5-chips-marcadores');
  chipsM.innerHTML = (est.top_marcadores || []).slice(0, 18).map(item =>
    `<span class="badge badge-azul">"${escHtml(item.frase)}" <strong>×${item.frecuencia}</strong></span>`
  ).join('') || '<span style="font-size:0.8rem;color:var(--muted)">Sin marcadores detectados</span>';

  const chipsN = document.getElementById('p5-chips-ngramas');
  chipsN.innerHTML = (est.frases_frecuentes || []).slice(0, 15).map(item =>
    `<span class="badge">"${escHtml(item.frase)}" <strong>×${item.frecuencia}</strong></span>`
  ).join('');

  document.getElementById('p5-inp-muletillas').value = (est.muletillas_personalizadas || []).join(', ');
}

document.getElementById('arch-p5-extra').onchange = e => {
  const n = e.target.files.length;
  document.getElementById('p5-lbl-extra').textContent = n ? `${n} archivo(s) listos para sumar al entrenamiento` : 'Ningún archivo extra seleccionado';
};

document.getElementById('btn-p5-entrenar').onclick = async () => {
  const btn = document.getElementById('btn-p5-entrenar');
  const resDiv = document.getElementById('p5-res-entrenar');
  btn.disabled = true;
  resDiv.innerHTML = '⏳ Analizando tus transcripciones y entrenando modelo de estilo…';

  try {
    const selecciones = [];
    document.querySelectorAll('.chk-ses-train:checked').forEach(chk => {
      const sid = chk.value;
      const selSpk = document.querySelector(`.sel-spk-train[data-sid="${sid}"]`);
      selecciones.push({ sesion_id: sid, hablante: selSpk ? selSpk.value : 'SPK_1' });
    });

    const textosExtra = [];
    const filesExtra = document.getElementById('arch-p5-extra').files;
    for (const f of filesExtra) {
      const q = new URLSearchParams({ nombre: f.name });
      const rUp = await fetch('/api/subir_transcripcion?' + q.toString(), { method: 'POST', body: f });
      if (rUp.ok) {
        const dUp = await rUp.json();
        selecciones.push({ sesion_id: dUp.sesion_id, hablante: 'ACTIVOS' });
      }
    }

    const r = await fetch('/api/estilo_entrenar', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({ selecciones, textos_extra: textosExtra })
    });
    if (!r.ok) throw new Error(await r.text());
    const modelo = await r.json();
    renderModeloEstiloUI(modelo);
    resDiv.innerHTML = `<span style="color:var(--ok);font-weight:700">✔ ¡Modelo entrenado con éxito! Analizó ${modelo.metricas.total_intervenciones} intervenciones (${modelo.metricas.total_palabras} palabras) de tus transcripciones.</span>`;
  } catch (e) {
    resDiv.innerHTML = `<span style="color:var(--danger);font-weight:700">✘ ${escHtml(e.message)}</span>`;
  } finally {
    btn.disabled = false;
  }
};

document.getElementById('btn-p5-guardar-muletillas').onclick = async () => {
  const val = document.getElementById('p5-inp-muletillas').value;
  const lista = val.split(',').map(x => x.trim()).filter(Boolean);
  const r = await fetch('/api/estilo_guardar_ajustes', {
    method: 'POST',
    headers: {'Content-Type':'application/json'},
    body: JSON.stringify({ muletillas_personalizadas: lista })
  });
  if (r.ok) {
    alert('✔ Tus expresiones preferidas quedaron guardadas en el modelo.');
  }
};

document.getElementById('btn-p5-ejemplo-guion').onclick = () => {
  document.getElementById('p5-txt-guion').value =
    "A continuación veremos cómo trabajar con funciones y librerías en una nueva versión de Python. Es importante destacar que primero debemos verificar qué versión tienes instalada aquí en el entorno.\n\n" +
    "Observa que si realizaremos un proyecto grande, utilizaremos muchísimos módulos externos; por lo tanto, debes crear un entorno virtual para no mezclar las dependencias. Finalmente, definiremos la función principal y probaremos cómo funciona.";
};

document.getElementById('btn-p5-convertir').onclick = async () => {
  const guion = document.getElementById('p5-txt-guion').value.trim();
  if (!guion) {
    alert('Escribí o pegá primero un guión (o tocá "✨ Cargar guión de prueba").');
    return;
  }
  const btn = document.getElementById('btn-p5-convertir');
  btn.disabled = true;
  btn.textContent = '⏳ Convirtiendo guión a tu estilo…';

  try {
    const r = await fetch('/api/estilo_convertir_guion', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({
        guion,
        intensidad: document.getElementById('p5-intensidad').value,
        formato_salida: document.getElementById('p5-formato-out').value,
        usar_ollama: document.getElementById('p5-chk-ollama').checked
      })
    });
    if (!r.ok) throw new Error(await r.text());
    const d = await r.json();
    ultimoPromptAdn = d.prompt_adn || '';
    document.getElementById('p5-box-resultado').style.display = 'block';
    document.getElementById('p5-txt-salida').value = d.guion_convertido;
    document.getElementById('p5-info-out').textContent = `${d.palabras_salida} palabras · ~${fmtHMS(d.duracion_estimada_seg)} hablados a tu ritmo`;
    const ragDiv = document.getElementById('p5-rag-ejemplos');
    ragDiv.innerHTML = (d.ejemplos_referencia || []).map(ej =>
      `<div style="padding:6px 8px;background:#fff;border-left:3px solid #6366f1;border-radius:4px">"${escHtml(ej.slice(0, 220))}${ej.length > 220 ? '…' : ''}"</div>`
    ).join('') || '<span>Sin fragmentos de referencia</span>';
  } catch (e) {
    alert('Error al convertir guión: ' + e.message);
  } finally {
    btn.disabled = false;
    btn.textContent = '🪄 Convertir Guión a Mi Estilo de Hablar';
  }
};

document.getElementById('btn-p5-copiar-out').onclick = () => {
  const txt = document.getElementById('p5-txt-salida').value;
  navigator.clipboard.writeText(txt);
  alert('✔ Guión convertido copiado al portapapeles.');
};

document.getElementById('btn-p5-descargar-out').onclick = () => {
  const txt = document.getElementById('p5-txt-salida').value;
  const blob = new Blob([txt], { type: 'text/plain;charset=utf-8' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'guion_en_mi_estilo.txt';
  a.click();
};

document.getElementById('btn-p5-copiar-prompt').onclick = () => {
  if (!ultimoPromptAdn) return;
  navigator.clipboard.writeText(ultimoPromptAdn);
  alert('✔ Prompt con tu ADN de estilo y ejemplos reales copiado.');
};


// Inicialización automática: si hay una sesión guardada, dejarla pre-cargada
(async function init() {
  await cargarAudiosDisponibles();
  await cargarSesionesGuardadas();
  try {
    const r = await fetch('/api/sesiones_guardadas');
    const lista = await r.json();
    if (lista.length > 0) {
      const rSes = await fetch('/api/sesion?id=' + encodeURIComponent(lista[0].sesion_id));
      if (rSes.ok) {
        activarSesion(await rSes.json());
      }
    }
  } catch (e) {}
})();
</script>
</body>
</html>
"""


# ==============================================================================
# SERVIDOR HTTP CON SOPORTE DE RANGOS (SEEKING DE AUDIO) Y API JSON
# ==============================================================================

class Handler(BaseHTTPRequestHandler):
    def _send_json(self, code: int, data: dict | list) -> None:
        raw = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _send_text(self, code: int, text: str, ctype: str = "text/plain; charset=utf-8") -> None:
        raw = text.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _serve_audio_file(self, ruta: str, nombre: str) -> None:
        tam = os.path.getsize(ruta)
        ext = os.path.splitext(nombre)[1].lower()
        if ext == ".wav":
            ctype = "audio/wav"
        elif ext == ".zip":
            ctype = "application/zip"
        elif ext in (".mp4", ".mov", ".mkv"):
            ctype = "video/mp4"
        elif ext == ".webm":
            ctype = "video/webm"
        elif ext == ".srt":
            ctype = "application/x-subrip; charset=utf-8"
        else:
            ctype = "audio/mpeg"
        rango = self.headers.get("Range")
        if rango and rango.startswith("bytes="):
            try:
                partes = rango[6:].split("-")
                ini = int(partes[0]) if partes[0] else 0
                fin = int(partes[1]) if len(partes) > 1 and partes[1] else tam - 1
                fin = min(fin, tam - 1)
                if 0 <= ini <= fin:
                    largo = fin - ini + 1
                    with open(ruta, "rb") as f:
                        f.seek(ini)
                        datos = f.read(largo)
                    self.send_response(206)
                    self.send_header("Content-Type", ctype)
                    self.send_header("Accept-Ranges", "bytes")
                    self.send_header("Content-Range", f"bytes {ini}-{fin}/{tam}")
                    self.send_header("Content-Length", str(len(datos)))
                    self.end_headers()
                    self.wfile.write(datos)
                    return
            except Exception:
                pass

        with open(ruta, "rb") as f:
            datos = f.read()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(len(datos)))
        disp = "attachment" if ext in (".zip", ".srt") else "inline"
        self.send_header("Content-Disposition", f"{disp}; filename*=UTF-8''" + urllib.parse.quote(nombre))
        self.end_headers()
        self.wfile.write(datos)

    def do_GET(self):
        p = urllib.parse.urlparse(self.path)
        if p.path == "/":
            return self._send_text(200, HTML, "text/html; charset=utf-8")

        if p.path == "/api/git_estado":
            q = urllib.parse.parse_qs(p.query)
            if q.get("verificar", ["0"])[0] == "1":
                st = verificar_y_sincronizar_git(accion_automatica=False)
            else:
                with GIT_LOCK:
                    st = dict(GIT_ESTADO)
            return self._send_json(200, st)

        if p.path == "/api/audios_existentes":
            archivos = []
            if os.path.isdir(SALIDA):
                for f in os.listdir(SALIDA):
                    if f.lower().endswith((".mp3", ".wav", ".m4a", ".ogg", ".flac", ".aac", ".mp4", ".mkv", ".mov", ".webm", ".avi")):
                        rp = os.path.join(SALIDA, f)
                        if os.path.isfile(rp):
                            archivos.append({
                                "nombre": f,
                                "tipo": "video" if f.lower().endswith((".mp4", ".mkv", ".mov", ".webm", ".avi")) else "audio",
                                "bytes": os.path.getsize(rp),
                                "mtime": os.path.getmtime(rp)
                            })
            archivos.sort(key=lambda x: x["mtime"], reverse=True)
            return self._send_json(200, archivos)

        if p.path == "/api/sesiones_guardadas":
            lista = []
            with JOBS_LOCK:
                for sid, s in SESIONES.items():
                    lista.append({
                        "sesion_id": sid,
                        "nombre_original": s.get("nombre_original", sid),
                        "nombre_mp3": s.get("nombre_mp3", ""),
                        "duracion": s.get("duracion", 0),
                        "modelo": s.get("modelo", "whisper"),
                        "cantidad_segmentos": len(s.get("segmentos", [])),
                        "cantidad_hablantes": len(s.get("hablantes", [])),
                        "offset": float(s.get("offset", 0) or 0),
                        "sincronizado": bool(s.get("sincronizado", False)),
                    })
            return self._send_json(200, lista)

        if p.path == "/api/sesion":
            q = urllib.parse.parse_qs(p.query)
            sid = q.get("id", [""])[0]
            with JOBS_LOCK:
                s = SESIONES.get(sid)
            if not s:
                return self._send_text(404, "Sesión no encontrada")
            return self._send_json(200, {
                "sesion_id": s["sesion_id"],
                "nombre_original": s.get("nombre_original", ""),
                "nombre_mp3": s.get("nombre_mp3", ""),
                "duracion": s.get("duracion", 0),
                "modelo": s.get("modelo", ""),
                "hablantes": s.get("hablantes", []),
                "segmentos": s.get("segmentos", []),
                "rango_detectado": s.get("rango_detectado", {}),
                "offset": float(s.get("offset", 0) or 0),
                "sincronizado": bool(s.get("sincronizado", False)),
                "origen_media": s.get("origen_media", ""),
            })

        if p.path == "/api/voces_registradas":
            reg = _asegurar_estructura_grupos(cargar_registro_voces())
            resumen = []
            for nom, info in reg.get("perfiles", {}).items():
                resumen.append({
                    "nombre": nom,
                    "color": info.get("color", "#2563eb"),
                    "n_muestras": info.get("n_muestras", 1),
                    "actualizado_en": info.get("actualizado_en") or info.get("creado_en", "")
                })
            return self._send_json(200, {
                "perfiles": resumen,
                "grupos": listar_grupos()
            })

        if p.path == "/api/estado":
            q = urllib.parse.parse_qs(p.query)
            jid = q.get("id", [""])[0]
            with JOBS_LOCK:
                job = JOBS.get(jid)
                if not job:
                    return self._send_text(404, "Trabajo no encontrado")
                copia = dict(job)
            return self._send_json(200, copia)

        if p.path == "/api/estilo_estado":
            modelo = cargar_modelo_estilo()
            copia = {k: v for k, v in modelo.items() if k != "bloques_rag"}
            return self._send_json(200, copia)

        if p.path == "/api/estilo_dataset_jsonl":
            modelo = cargar_modelo_estilo()
            jsonl_txt = generar_dataset_finetuning_jsonl(modelo)
            raw = jsonl_txt.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/jsonl; charset=utf-8")
            self.send_header("Content-Disposition", 'attachment; filename="mi_estilo_finetuning.jsonl"')
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
            return

        if p.path == "/api/zip_partes":
            q = urllib.parse.parse_qs(p.query)
            nombres = [os.path.basename(n) for n in q.get("n", []) if n]
            rutas = [os.path.join(SALIDA, n) for n in nombres if os.path.isfile(os.path.join(SALIDA, n))]
            if not rutas:
                return self._send_text(404, "No se encontraron las partes.")
            base_zip = os.path.splitext(nombres[0])[0].split("_parte_")[0] + "_partes.zip"
            ruta_zip = os.path.join(TEMP_DIR, uuid.uuid4().hex[:8] + "_" + base_zip)
            with zipfile.ZipFile(ruta_zip, "w", zipfile.ZIP_STORED) as zf:
                for r in rutas:
                    zf.write(r, os.path.basename(r))
            try:
                tam = os.path.getsize(ruta_zip)
                self.send_response(200)
                self.send_header("Content-Type", "application/zip")
                self.send_header("Content-Length", str(tam))
                self.send_header("Content-Disposition", "attachment; filename*=UTF-8''" + urllib.parse.quote(base_zip))
                self.end_headers()
                with open(ruta_zip, "rb") as f:
                    shutil.copyfileobj(f, self.wfile, 1 << 20)
            finally:
                try:
                    os.remove(ruta_zip)
                except OSError:
                    pass
            return

        if p.path.startswith("/audios/"):
            nombre = os.path.basename(urllib.parse.unquote(p.path[8:]))
            ruta = os.path.join(SALIDA, nombre)
            if os.path.isfile(ruta):
                return self._serve_audio_file(ruta, nombre)

        self._send_text(404, "No encontrado")

    def do_POST(self):
        p = urllib.parse.urlparse(self.path)

        if p.path == "/api/git_sincronizar":
            largo = int(self.headers.get("Content-Length", 0) or 0)
            body = {}
            if largo > 0:
                try:
                    body = json.loads(self.rfile.read(largo).decode("utf-8"))
                except Exception:
                    body = {}
            accion = body.get("accion", "auto")
            url_remote = (body.get("remote_url") or "").strip()

            if accion == "configurar_remote" and url_remote:
                ejecutar_git("remote", "remove", "origin")
                code_a, _, err_a = ejecutar_git("remote", "add", "origin", url_remote)
                if code_a != 0:
                    return self._send_text(400, f"Error al configurar remote: {err_a}")
                ejecutar_git("add", ".gitignore", "app.py", "README.md", "Extraer audio.bat", "voces_registradas.json")
                ejecutar_git("commit", "-m", "chore: guardar estado local antes de sincronizar con remote")

            if accion == "pull":
                with GIT_LOCK:
                    code_p, _, err_p = ejecutar_git("pull", "--rebase", "origin", GIT_ESTADO.get("rama", "main"))
                    if code_p == 0:
                        verificar_y_sincronizar_git(accion_automatica=False)
                        return self._send_json(200, {"ok": True, "mensaje": "Pull realizado con éxito desde GitHub."})
                    else:
                        return self._send_text(500, f"Error en git pull: {err_p}")

            if accion == "push":
                with GIT_LOCK:
                    ejecutar_git("add", ".gitignore", "app.py", "README.md", "Extraer audio.bat", "voces_registradas.json")
                    ejecutar_git("commit", "-m", "chore: actualizar cambios locales")
                    code_p, _, err_p = ejecutar_git("push", "origin", GIT_ESTADO.get("rama", "main"))
                    if code_p == 0:
                        verificar_y_sincronizar_git(accion_automatica=False)
                        return self._send_json(200, {"ok": True, "mensaje": "Push realizado con éxito a GitHub."})
                    else:
                        return self._send_text(500, f"Error en git push: {err_p}")

            res = verificar_y_sincronizar_git(accion_automatica=True)
            return self._send_json(200, res)

        if p.path == "/api/crear_grupo":
            largo = int(self.headers.get("Content-Length", 0) or 0)
            body = json.loads(self.rfile.read(largo).decode("utf-8")) if largo > 0 else {}
            nom = (body.get("nombre") or "").strip()
            if not nom:
                return self._send_text(400, "El nombre del grupo es obligatorio.")
            g = crear_o_actualizar_grupo(nom, body.get("categoria", ""), body.get("area", ""), body.get("subgrupo", ""))
            return self._send_json(200, {"ok": True, "grupo": g, "grupos": listar_grupos()})

        if p.path == "/api/subir_listado_nombres":
            q = urllib.parse.parse_qs(p.query)
            nombre_arch = os.path.basename(q.get("nombre", ["nombres.txt"])[0])
            grupo = q.get("grupo", ["General"])[0]
            reemplazar = q.get("reemplazar", ["0"])[0] == "1"
            largo = int(self.headers.get("Content-Length", 0) or 0)
            datos_bytes = self.rfile.read(largo) if largo > 0 else b""
            if not datos_bytes:
                return self._send_text(400, "No se enviaron datos.")
            try:
                extraidos = extraer_nombres_desde_archivo(datos_bytes, nombre_arch)
                lista_final = agregar_nombres_al_listado(extraidos, grupo=grupo, reemplazar=reemplazar)
                return self._send_json(200, {"agregados": len(extraidos), "nombres": lista_final})
            except Exception as e:
                return self._send_text(400, f"Error leyendo el listado de nombres: {e}")

        if p.path == "/api/eliminar_nombre_listado":
            largo = int(self.headers.get("Content-Length", 0) or 0)
            body = json.loads(self.rfile.read(largo).decode("utf-8"))
            nom_borrar = (body.get("nombre") or "").strip().lower()
            grupo = (body.get("grupo") or "General").strip()
            reg = _asegurar_estructura_grupos(cargar_registro_voces())
            if grupo in reg["grupos"]:
                reg["grupos"][grupo]["nombres"] = [x for x in reg["grupos"][grupo].get("nombres", []) if x.lower() != nom_borrar]
            reg["listado_nombres"] = [x for x in reg.get("listado_nombres", []) if x.lower() != nom_borrar]
            guardar_registro_voces(reg)
            return self._send_json(200, {"nombres": obtener_listado_nombres(grupo)})

        if p.path == "/api/subir_transcripcion":
            q = urllib.parse.parse_qs(p.query)
            nombre_arch = os.path.basename(q.get("nombre", ["transcripcion.txt"])[0])
            nombre_mp3 = os.path.basename(q.get("nombre_mp3", [""])[0]) if q.get("nombre_mp3", [""])[0] else ""
            desde_str = (q.get("desde", [""])[0] or "").strip()
            hasta_str = (q.get("hasta", [""])[0] or "").strip()
            reiniciar_rel = q.get("reiniciar_reloj", ["0"])[0] == "1"
            largo = int(self.headers.get("Content-Length", 0) or 0)
            datos_bytes = self.rfile.read(largo) if largo > 0 else b""
            if not datos_bytes:
                return self._send_text(400, "El archivo de transcripción está vacío.")
            try:
                t_desde = parsear_tiempo(desde_str) if desde_str else None
                t_hasta = parsear_tiempo(hasta_str) if hasta_str else None
                parsed = parsear_transcripcion_externa(datos_bytes, nombre_arch)
                if t_desde is not None or t_hasta is not None:
                    segs_rec, _, dur_rec = recortar_segmentos_por_tiempo(
                        parsed["segmentos"], t_desde, t_hasta, reiniciar_reloj=reiniciar_rel
                    )
                    if not segs_rec:
                        return self._send_text(400, "No quedaron fragmentos dentro del minuto y segundo indicados.")
                    parsed["segmentos"] = segs_rec
                    parsed["duracion"] = dur_rec
                    nomes_prev = {h["id"]: h["nombre"] for h in parsed.get("hablantes", [])}
                    parsed["hablantes"] = _resumen_hablantes(segs_rec, nombres_previos=nomes_prev)
            except Exception as e:
                return self._send_text(400, f"Error leyendo la transcripción: {e}")
            if not parsed["segmentos"]:
                return self._send_text(400, "No se encontraron líneas en la transcripción.")

            sid = uuid.uuid4().hex[:12]
            ruta_mp3 = os.path.join(SALIDA, nombre_mp3) if nombre_mp3 else ""
            sesion_data = {
                "sesion_id": sid,
                "nombre_original": nombre_arch,
                "nombre_base": os.path.splitext(nombre_arch)[0],
                "nombre_mp3": nombre_mp3,
                "ruta_mp3": ruta_mp3,
                "ruta_wav16k": "",
                "cal": "192k",
                "mono": True,
                "duracion": parsed["duracion"],
                "modelo": f"Importado ({parsed['formato_detectado']})",
                "formato_detectado": parsed["formato_detectado"],
                "segmentos": parsed["segmentos"],
                "hablantes": parsed["hablantes"],
                "X": None,
            }
            with JOBS_LOCK:
                SESIONES[sid] = sesion_data
            guardar_sesion_disco(sesion_data)
            return self._send_json(200, {
                "sesion_id": sid,
                "nombre_original": nombre_arch,
                "nombre_mp3": nombre_mp3,
                "duracion": parsed["duracion"],
                "modelo": sesion_data["modelo"],
                "formato_detectado": parsed["formato_detectado"],
                "hablantes": parsed["hablantes"],
                "segmentos": parsed["segmentos"],
            })

        if p.path == "/api/auto_matchear_presentaciones":
            largo = int(self.headers.get("Content-Length", 0) or 0)
            body = json.loads(self.rfile.read(largo).decode("utf-8"))
            sid = body.get("sesion_id", "")
            grupo = body.get("grupo", "General")
            with JOBS_LOCK:
                sesion = SESIONES.get(sid)
            if not sesion:
                return self._send_text(404, "Sesión no encontrada")

            listado = obtener_listado_nombres(grupo)
            hallazgos = detectar_presentaciones_en_segmentos(sesion["segmentos"], listado)
            # Asignar los nombres detectados a los hablantes correspondientes
            mapa_spk_nom = {}
            for h_item in hallazgos:
                mapa_spk_nom[h_item["spk_id"]] = h_item["nombre_detectado"]

            for h in sesion["hablantes"]:
                if h["id"] in mapa_spk_nom:
                    h["nombre"] = mapa_spk_nom[h["id"]]
                    h["auto_etiquetado"] = True
                    # Si hay matriz acústica X, registrar su huella en el grupo
                    X = sesion.get("X")
                    if X is not None:
                        idx_s = [i for i, s in enumerate(sesion["segmentos"]) if s.get("hablante") == h["id"]]
                        if idx_s and len(X) > max(idx_s):
                            registrar_voz_en_grupo(grupo, h["nombre"], np.mean(X[idx_s], axis=0), color=h.get("color", "#2563eb"), origen="presentacion")

            guardar_sesion_disco(sesion)
            return self._send_json(200, {
                "hallazgos": hallazgos,
                "segmentos": sesion["segmentos"],
                "hablantes": sesion["hablantes"]
            })

        if p.path == "/api/sincronizar_tactiq_voces":
            q = urllib.parse.parse_qs(p.query)
            sid = q.get("sesion_id", [""])[0]
            nombre_arch = os.path.basename(q.get("nombre", ["tactiq.txt"])[0])
            grupo = q.get("grupo", ["General"])[0]
            largo = int(self.headers.get("Content-Length", 0) or 0)
            datos_bytes = self.rfile.read(largo) if largo > 0 else b""
            with JOBS_LOCK:
                sesion = SESIONES.get(sid)
            if not sesion:
                return self._send_text(404, "Sesión no encontrada")
            try:
                parsed_t = parsear_transcripcion_externa(datos_bytes, nombre_arch)
                listado = obtener_listado_nombres(grupo)
                res_sync = sincronizar_y_matchear_con_tactiq(
                    sesion["segmentos"], parsed_t["segmentos"], parsed_t["hablantes"], listado
                )
                mapeo = res_sync.get("mapeo_hablantes", {})
                for h in sesion["hablantes"]:
                    if h["id"] in mapeo:
                        h["nombre"] = mapeo[h["id"]]
                        h["auto_etiquetado"] = True
                        X = sesion.get("X")
                        if X is not None:
                            idx_s = [i for i, s in enumerate(sesion["segmentos"]) if s.get("hablante") == h["id"]]
                            if idx_s and len(X) > max(idx_s):
                                registrar_voz_en_grupo(grupo, h["nombre"], np.mean(X[idx_s], axis=0), color=h.get("color", "#2563eb"), origen="tactiq")
                guardar_sesion_disco(sesion)
                return self._send_json(200, {
                    **res_sync,
                    "segmentos": sesion["segmentos"],
                    "hablantes": sesion["hablantes"]
                })
            except Exception as e:
                return self._send_text(400, f"Error al sincronizar con Tactiq: {e}")

        if p.path in ("/api/publicar_subir_video", "/api/subir_medio"):
            q = urllib.parse.parse_qs(p.query)
            original = os.path.basename(q.get("nombre", ["video.mp4"])[0])
            base_v, ext_v = os.path.splitext(original)
            permitidas = (".mp4", ".mkv", ".mov", ".webm", ".avi", ".mp3", ".wav", ".m4a", ".ogg", ".flac", ".aac")
            if ext_v.lower() not in permitidas:
                return self._send_text(400, "Formato no admitido. Subí un audio (MP3, WAV, M4A, OGG) o un video (MP4, MKV, MOV, WEBM).")
            ext_v = ext_v.lower()
            largo = int(self.headers.get("Content-Length", 0) or 0)
            if largo <= 0:
                return self._send_text(400, "No llegó ningún video.")
            nombre_dest = nombre_unico(SALIDA, base_v, ext_v)
            with open(os.path.join(SALIDA, nombre_dest), "wb") as f:
                restante = largo
                while restante > 0:
                    trozo = self.rfile.read(min(1 << 20, restante))
                    if not trozo:
                        break
                    f.write(trozo)
                    restante -= len(trozo)
            return self._send_json(200, {"nombre_video": nombre_dest, "nombre": nombre_dest})

        if p.path == "/api/publicar_render":
            largo = int(self.headers.get("Content-Length", 0) or 0)
            body = json.loads(self.rfile.read(largo).decode("utf-8")) if largo > 0 else {}
            if not body.get("sesion_id") or not body.get("nombre_video"):
                return self._send_text(400, "Falta el video o el análisis de voces.")
            job_id = uuid.uuid4().hex[:12]
            with JOBS_LOCK:
                JOBS[job_id] = {"id": job_id, "estado": "procesando", "progreso": 2,
                                "mensaje": "Preparando…", "resultado": None, "error": None}
            threading.Thread(target=_worker_publicar, args=(job_id, body), daemon=True).start()
            return self._send_json(200, {"job_id": job_id})

        if p.path == "/api/sesion_guardar":
            # Guarda las correcciones hechas en la app: hablante de cada línea, texto, líneas excluidas y nombres
            largo = int(self.headers.get("Content-Length", 0) or 0)
            body = json.loads(self.rfile.read(largo).decode("utf-8")) if largo > 0 else {}
            sid = body.get("sesion_id", "")
            with JOBS_LOCK:
                sesion = SESIONES.get(sid)
            if not sesion:
                return self._send_text(404, "Transcripción no encontrada")
            segs_in = body.get("segmentos")
            if isinstance(segs_in, list) and len(segs_in) == len(sesion.get("segmentos", [])):
                for s_srv, s_cli in zip(sesion["segmentos"], segs_in):
                    if "hablante" in s_cli and s_cli["hablante"]:
                        s_srv["hablante"] = str(s_cli["hablante"])
                    if "texto" in s_cli:
                        s_srv["texto"] = str(s_cli["texto"])
                    if s_cli.get("override_mute") is True:
                        s_srv["override_mute"] = True
                    else:
                        s_srv.pop("override_mute", None)
            previos = {h["id"]: h for h in sesion.get("hablantes", [])}
            for h in body.get("hablantes") or []:
                if h.get("id") in previos:
                    previos[h["id"]]["nombre"] = h.get("nombre") or previos[h["id"]]["nombre"]
                else:
                    previos[h["id"]] = {"id": h["id"], "nombre": h.get("nombre") or h["id"], "color": h.get("color", "#64748b")}
            nombres = {k: v.get("nombre") for k, v in previos.items()}
            colores = {k: v.get("color") for k, v in previos.items()}
            nuevos = _resumen_hablantes(sesion["segmentos"], nombres_previos=nombres)
            for h in nuevos:
                if colores.get(h["id"]):
                    h["color"] = colores[h["id"]]
                if previos.get(h["id"], {}).get("auto_etiquetado"):
                    h["auto_etiquetado"] = True
            sesion["hablantes"] = nuevos
            guardar_sesion_disco(sesion)
            return self._send_json(200, {"ok": True, "hablantes": nuevos})

        if p.path == "/api/sesion_vincular":
            largo = int(self.headers.get("Content-Length", 0) or 0)
            body = json.loads(self.rfile.read(largo).decode("utf-8")) if largo > 0 else {}
            sid = body.get("sesion_id", "")
            with JOBS_LOCK:
                sesion = SESIONES.get(sid)
            if not sesion:
                return self._send_text(404, "Transcripción no encontrada")
            if "nombre_media" in body:
                nom = os.path.basename(body.get("nombre_media") or "")
                if nom and not os.path.isfile(os.path.join(SALIDA, nom)):
                    return self._send_text(404, f"No está en audios/: {nom}")
                if nom != sesion.get("nombre_mp3", ""):
                    sesion["origen_media"] = "vinculado"
                sesion["nombre_mp3"] = nom
                sesion["ruta_mp3"] = os.path.join(SALIDA, nom) if nom else ""
            if "offset" in body:
                sesion["offset"] = round(float(body.get("offset") or 0), 2)
            if "sincronizado" in body:
                sesion["sincronizado"] = bool(body.get("sincronizado"))
            guardar_sesion_disco(sesion)
            return self._send_json(200, {"ok": True, "nombre_mp3": sesion.get("nombre_mp3", ""),
                                         "offset": sesion.get("offset", 0), "sincronizado": sesion.get("sincronizado", False),
                                         "origen_media": sesion.get("origen_media", "")})

        if p.path == "/api/iniciar":
            q = urllib.parse.parse_qs(p.query)
            original = os.path.basename(q.get("nombre", ["video.mp4"])[0])
            usar_existente = q.get("usar_mp3_existente", ["0"])[0] == "1"

            try:
                parsear_tiempo(q.get("desde", [""])[0])
                parsear_tiempo(q.get("hasta", [""])[0])
            except ValueError as e:
                return self._send_text(400, str(e))

            largo = int(self.headers.get("Content-Length", 0) or 0)
            if usar_existente:
                if largo > 0:
                    self.rfile.read(largo)
                ruta_entrada = os.path.join(SALIDA, original)
                if not os.path.isfile(ruta_entrada):
                    return self._send_text(404, f"No se encontró el archivo en /audios: {original}")
                es_temporal = False
            else:
                ext = os.path.splitext(original)[1].lower() or ".mp4"
                with tempfile.NamedTemporaryFile(suffix=ext, delete=False, dir=TEMP_DIR) as tmp:
                    restante = largo
                    while restante > 0:
                        trozo = self.rfile.read(min(1 << 20, restante))
                        if not trozo:
                            break
                        tmp.write(trozo)
                        restante -= len(trozo)
                    ruta_entrada = tmp.name
                es_temporal = True

            params = {
                "cal": q.get("cal", ["192k"])[0],
                "desde": q.get("desde", [""])[0],
                "hasta": q.get("hasta", [""])[0],
                "frase_ini": q.get("frase_ini", [""])[0],
                "frase_fin": q.get("frase_fin", [""])[0],
                "mb": q.get("mb", ["0"])[0] or 0,
                "mono": q.get("mono", ["0"])[0] == "1",
                "modelo": q.get("modelo", ["small"])[0],
                "idioma": q.get("idioma", ["es"])[0],
                "hablantes": q.get("hablantes", ["0"])[0] or 0,
                "solo_extraer": q.get("solo_extraer", ["0"])[0] == "1",
                "modo_p1": q.get("modo_p1", ["mp3"])[0],
                "usar_mp3_existente": usar_existente,
                "split_modo": q.get("split_modo", [""])[0],
                "split_puntos": q.get("split_puntos", [""])[0],
                "split_duracion": q.get("split_duracion", [""])[0],
                "split_exacto": q.get("split_exacto", ["0"])[0] == "1",
            }

            job_id = uuid.uuid4().hex[:12]
            with JOBS_LOCK:
                JOBS[job_id] = {
                    "id": job_id,
                    "estado": "procesando",
                    "progreso": 4,
                    "mensaje": "Iniciando procesamiento…",
                    "resultado": None,
                    "error": None,
                }

            hilo = threading.Thread(
                target=_worker_analizar,
                args=(job_id, ruta_entrada, es_temporal, original, params),
                daemon=True
            )
            hilo.start()
            return self._send_json(200, {"job_id": job_id})

        if p.path == "/api/buscar_frases":
            largo = int(self.headers.get("Content-Length", 0) or 0)
            body = json.loads(self.rfile.read(largo).decode("utf-8"))
            frase_ini = (body.get("frase_ini") or "").strip()
            frase_fin = (body.get("frase_fin") or "").strip()
            segmentos = body.get("segmentos") or []

            match_ini = None
            match_fin = None
            t_min = 0.0
            if frase_ini:
                m_ini = buscar_frases_en_segmentos(segmentos, frase_ini, desde_seg=0.0)
                if m_ini:
                    match_ini = m_ini[0]
                    t_min = float(match_ini["frase_inicio"])
            if frase_fin:
                m_fin = buscar_frases_en_segmentos(segmentos, frase_fin, desde_seg=t_min)
                if m_fin:
                    match_fin = m_fin[0]

            return self._send_json(200, {"match_ini": match_ini, "match_fin": match_fin})

        if p.path == "/api/reagrupar":
            largo = int(self.headers.get("Content-Length", 0) or 0)
            body = json.loads(self.rfile.read(largo).decode("utf-8"))
            sid = body.get("sesion_id", "")
            num_hablantes = int(body.get("num_hablantes", 0) or 0)
            with JOBS_LOCK:
                sesion = SESIONES.get(sid)
            if not sesion:
                return self._send_text(404, "Sesión no encontrada")

            segmentos, hablantes = agrupar_hablantes(
                sesion["segmentos"], sesion.get("X"), num_hablantes=num_hablantes
            )
            sesion["segmentos"] = segmentos
            sesion["hablantes"] = hablantes
            guardar_sesion_disco(sesion)
            return self._send_json(200, {"segmentos": segmentos, "hablantes": hablantes})

        if p.path == "/api/guardar_voz_registro":
            largo = int(self.headers.get("Content-Length", 0) or 0)
            body = json.loads(self.rfile.read(largo).decode("utf-8"))
            sid = body.get("sesion_id", "")
            spk_id = body.get("spk_id", "")
            nombre = (body.get("nombre") or "").strip()
            color = body.get("color", "")
            grupo = (body.get("grupo") or "General").strip()

            if not nombre:
                return self._send_text(400, "El nombre del hablante es obligatorio")

            with JOBS_LOCK:
                sesion = SESIONES.get(sid)
            if not sesion:
                return self._send_text(404, "Sesión no encontrada")

            X = sesion.get("X")
            segmentos = sesion.get("segmentos", [])
            indices_spk = [i for i, s in enumerate(segmentos) if s.get("hablante") == spk_id]
            vector_voz = None
            if indices_spk and X is not None and len(X) > max(indices_spk):
                vector_voz = np.mean(X[indices_spk], axis=0)

            try:
                registrar_voz_en_grupo(grupo, nombre, vector_voz, color=color or "#2563eb", origen="manual")
            except Exception as e:
                return self._send_text(500, str(e))

            for h in sesion.get("hablantes", []):
                if h["id"] == spk_id:
                    h["nombre"] = nombre
                    h["auto_etiquetado"] = True
                    if color:
                        h["color"] = color
            guardar_sesion_disco(sesion)
            return self._send_json(200, {"ok": True})

        if p.path == "/api/eliminar_voz_registro":
            largo = int(self.headers.get("Content-Length", 0) or 0)
            body = json.loads(self.rfile.read(largo).decode("utf-8"))
            nombre = (body.get("nombre") or "").strip()
            if not nombre:
                return self._send_text(400, "Nombre requerido")
            ok = eliminar_voz_perfil(nombre)
            return self._send_json(200, {"ok": ok})

        if p.path == "/api/recortar_transcripcion_sesion":
            largo = int(self.headers.get("Content-Length", 0) or 0)
            body = json.loads(self.rfile.read(largo).decode("utf-8")) if largo > 0 else {}
            sid = body.get("sesion_id", "")
            desde_str = (body.get("desde") or "").strip()
            hasta_str = (body.get("hasta") or "").strip()
            reiniciar_reloj = bool(body.get("reiniciar_reloj", False))

            with JOBS_LOCK:
                sesion = SESIONES.get(sid)
            if not sesion:
                return self._send_text(404, "Sesión no encontrada")

            try:
                t_desde = parsear_tiempo(desde_str) if desde_str else None
                t_hasta = parsear_tiempo(hasta_str) if hasta_str else None
            except ValueError as e:
                return self._send_text(400, str(e))

            if t_desde is None and t_hasta is None:
                return self._send_text(400, "Indicá al menos el minuto:segundo Desde o Hasta para recortar.")

            nuevos_segs, indices_orig, nueva_dur = recortar_segmentos_por_tiempo(
                sesion.get("segmentos", []), t_desde, t_hasta, reiniciar_reloj=reiniciar_reloj
            )
            if not nuevos_segs:
                return self._send_text(400, "No hay fragmentos de transcripción dentro de ese rango de minutos y segundos.")

            X_old = sesion.get("X")
            if X_old is not None and isinstance(X_old, np.ndarray) and len(X_old) > max(indices_orig, default=-1):
                sesion["X"] = X_old[indices_orig]
            else:
                sesion["X"] = None

            nombres_prev = {h["id"]: h["nombre"] for h in sesion.get("hablantes", [])}
            sil_prev = {h["id"]: bool(h.get("silenciado")) for h in sesion.get("hablantes", [])}
            sesion["segmentos"] = nuevos_segs
            sesion["duracion"] = nueva_dur
            sesion["hablantes"] = _resumen_hablantes(nuevos_segs, nombres_previos=nombres_prev, silenciados_previos=sil_prev)
            guardar_sesion_disco(sesion)
            return self._send_json(200, {
                "ok": True,
                "duracion": nueva_dur,
                "segmentos": sesion["segmentos"],
                "hablantes": sesion["hablantes"]
            })

        if p.path == "/api/unir_subtitulos_video":
            q = urllib.parse.parse_qs(p.query)
            nombre_vid = os.path.basename(q.get("nombre_video", ["video.mp4"])[0])
            modo_sub = q.get("modo", ["soft"])[0]
            sid = q.get("sesion_id", [""])[0]
            offset_seg = float(q.get("offset", ["0"])[0] or 0.0)
            inc_nom = q.get("inc_nom", ["1"])[0] == "1"
            inc_sil = q.get("inc_sil", ["0"])[0] == "1"
            desde_str = (q.get("desde", [""])[0] or "").strip()
            hasta_str = (q.get("hasta", [""])[0] or "").strip()

            try:
                t_desde_v = parsear_tiempo(desde_str) if desde_str else None
                t_hasta_v = parsear_tiempo(hasta_str) if hasta_str else None
            except ValueError as e:
                return self._send_text(400, str(e))

            with JOBS_LOCK:
                sesion = SESIONES.get(sid)
            if not sesion and modo_sub != "solo_recortar_video":
                return self._send_text(404, "Sesión de transcripción no encontrada")

            largo = int(self.headers.get("Content-Length", 0) or 0)
            video_es_temporal = largo > 0
            if largo <= 0:
                ruta_video_in = os.path.join(SALIDA, nombre_vid)
                if not os.path.isfile(ruta_video_in):
                    return self._send_text(400, "No se encontró el video en audios/. Vinculalo en Transcripción.")
            else:
                ext_v = os.path.splitext(nombre_vid)[1].lower() or ".mp4"
                with tempfile.NamedTemporaryFile(suffix=ext_v, delete=False, dir=TEMP_DIR) as tmp_v:
                    restante = largo
                    while restante > 0:
                        trozo = self.rfile.read(min(1 << 20, restante))
                        if not trozo:
                            break
                        tmp_v.write(trozo)
                        restante -= len(trozo)
                    ruta_video_in = tmp_v.name

            base_v = os.path.splitext(nombre_vid)[0]
            suf_t = ""
            if desde_str or hasta_str:
                suf_t = f"_{(desde_str or '00-00').replace(':','-')}_a_{(hasta_str or 'fin').replace(':','-')}"

            if modo_sub == "solo_recortar_video":
                nombre_out = nombre_unico(SALIDA, f"{base_v}{suf_t}_recortado", ".mp4")
                ruta_out = os.path.join(SALIDA, nombre_out)
                try:
                    recortar_video_ffmpeg(ruta_video_in, ruta_out, desde=desde_str, hasta=hasta_str)
                except Exception as e:
                    return self._send_text(500, str(e))
                finally:
                    if video_es_temporal and os.path.exists(ruta_video_in):
                        try:
                            os.remove(ruta_video_in)
                        except OSError:
                            pass
                return self._send_json(200, {"ok": True, "nombre_archivo": nombre_out})

            bloque_00 = q.get("bloque_00", ["1"])[0] == "1"
            max_chars_q = int(q.get("max_chars", ["84"])[0] or 84)
            max_dur_q = float(q.get("max_dur", ["6.0"])[0] or 6.0)
            mapa_nom = {h["id"]: h["nombre"] for h in sesion.get("hablantes", [])}
            mapa_sil = {h["id"]: bool(h.get("silenciado")) for h in sesion.get("hablantes", [])}
            bloques_pro = construir_subtitulos_profesionales(
                sesion.get("segmentos", []),
                mapa_nom=mapa_nom,
                mapa_sil=mapa_sil,
                offset_seg=offset_seg,
                inc_nom=inc_nom,
                inc_sil=inc_sil,
                agregar_bloque_00=bloque_00,
                t_desde_v=t_desde_v,
                t_hasta_v=t_hasta_v,
                max_chars_sub=max_chars_q,
                min_dur_seg=1.2,
                max_dur_seg=max_dur_q,
            )
            lineas_srt = []
            for idx_b, b in enumerate(bloques_pro, start=1):
                ms_i = int(round(b["ini"] * 1000))
                ms_f = int(round(b["fin"] * 1000))
                ts_i = f"{ms_i//3600000:02d}:{(ms_i%3600000)//60000:02d}:{(ms_i%60000)//1000:02d},{ms_i%1000:03d}"
                ts_f = f"{ms_f//3600000:02d}:{(ms_f%3600000)//60000:02d}:{(ms_f%60000)//1000:02d},{ms_f%1000:03d}"
                lineas_srt.append(f"{idx_b}\n{ts_i} --> {ts_f}\n{b['texto_srt']}\n")

            ruta_srt_tmp = os.path.join(TEMP_DIR, f"sub_{uuid.uuid4().hex[:8]}.srt")
            with open(ruta_srt_tmp, "w", encoding="utf-8") as fsrt:
                fsrt.write("\n".join(lineas_srt))

            nombre_out = nombre_unico(SALIDA, f"{base_v}{suf_t}_subtitulado", ".mp4")
            ruta_out = os.path.join(SALIDA, nombre_out)
            ruta_vid_base = ruta_video_in
            ruta_vid_rec_tmp = ""

            try:
                if desde_str or hasta_str:
                    ruta_vid_rec_tmp = os.path.join(TEMP_DIR, f"vrec_{uuid.uuid4().hex[:8]}.mp4")
                    recortar_video_ffmpeg(ruta_video_in, ruta_vid_rec_tmp, desde=desde_str, hasta=hasta_str)
                    ruta_vid_base = ruta_vid_rec_tmp

                if modo_sub == "burn":
                    ruta_esc = ruta_srt_tmp.replace("\\", "/").replace(":", "\\:")
                    cmd = ["ffmpeg", "-y", "-i", ruta_vid_base, "-vf", f"subtitles='{ruta_esc}'", "-c:a", "copy", ruta_out]
                else:
                    cmd = ["ffmpeg", "-y", "-i", ruta_vid_base, "-i", ruta_srt_tmp, "-map", "0:v:0", "-map", "0:a?", "-map", "1:0",
                           "-c:v", "copy", "-c:a", "copy", "-c:s", "mov_text", "-metadata:s:s:0", "language=spa", ruta_out]
                r_ff = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
                if r_ff.returncode != 0:
                    return self._send_text(500, f"Error de ffmpeg: {r_ff.stderr[-300:]}")
            finally:
                for rp in ((ruta_video_in if video_es_temporal else ""), ruta_vid_rec_tmp, ruta_srt_tmp):
                    if rp and os.path.exists(rp):
                        try:
                            os.remove(rp)
                        except OSError:
                            pass

            return self._send_json(200, {"ok": True, "nombre_archivo": nombre_out})

        if p.path == "/api/estilo_entrenar":
            largo = int(self.headers.get("Content-Length", 0) or 0)
            body = json.loads(self.rfile.read(largo).decode("utf-8")) if largo > 0 else {}
            selecciones = body.get("selecciones") or []
            textos_extra = body.get("textos_extra") or []

            oraciones = []
            duraciones = []
            fuentes = []
            with JOBS_LOCK:
                for item in selecciones:
                    sid = item.get("sesion_id", "")
                    spk_filtro = item.get("hablante", "SPK_1")
                    ses = SESIONES.get(sid)
                    if not ses:
                        continue
                    fuentes.append(ses.get("nombre_original", sid))
                    mapa_sil = {h["id"]: bool(h.get("silenciado")) for h in ses.get("hablantes", [])}
                    for s in ses.get("segmentos", []):
                        if spk_filtro == "ACTIVOS":
                            if mapa_sil.get(s.get("hablante"), False):
                                continue
                        elif s.get("hablante") != spk_filtro:
                            continue
                        oraciones.append(s.get("texto", ""))
                        duraciones.append(max(0.5, float(s.get("fin", 0)) - float(s.get("inicio", 0))))

            for t_ex in textos_extra:
                for linea in str(t_ex).splitlines():
                    if linea.strip():
                        oraciones.append(linea.strip())

            try:
                modelo = entrenar_modelo_estilo_ml(oraciones, duraciones if duraciones else None, fuentes)
                copia = {k: v for k, v in modelo.items() if k != "bloques_rag"}
                return self._send_json(200, copia)
            except Exception as e:
                return self._send_text(400, str(e))

        if p.path == "/api/estilo_guardar_ajustes":
            largo = int(self.headers.get("Content-Length", 0) or 0)
            body = json.loads(self.rfile.read(largo).decode("utf-8")) if largo > 0 else {}
            modelo = cargar_modelo_estilo()
            modelo["muletillas_personalizadas"] = body.get("muletillas_personalizadas", [])
            guardar_modelo_estilo(modelo)
            return self._send_json(200, {"ok": True})

        if p.path == "/api/estilo_convertir_guion":
            largo = int(self.headers.get("Content-Length", 0) or 0)
            body = json.loads(self.rfile.read(largo).decode("utf-8")) if largo > 0 else {}
            try:
                res = convertir_guion_a_mi_estilo(
                    body.get("guion", ""),
                    intensidad=body.get("intensidad", "equilibrado"),
                    formato_salida=body.get("formato_salida", "fluido"),
                    usar_ollama=bool(body.get("usar_ollama", False))
                )
                return self._send_json(200, res)
            except Exception as e:
                return self._send_text(400, str(e))

        if p.path == "/api/exportar_mp3":
            largo = int(self.headers.get("Content-Length", 0) or 0)
            body = json.loads(self.rfile.read(largo).decode("utf-8"))
            sid = body.get("sesion_id", "")
            modo = body.get("modo", "aislar")
            if modo not in ("aislar", "silenciar", "recortar", "rango_directo"):
                modo = "aislar"
            formato = body.get("formato", "mp3")
            if formato not in ("mp3", "wav", "zip"):
                formato = "mp3"
            max_minutos = float(body.get("max_minutos", 0) or 0)
            min_dur_seg = float(body.get("min_dur_seg", 0) or 0)
            pausa_seg = float(body.get("pausa_seg", 0.3) or 0.3)
            rango_inicio = body.get("rango_inicio")
            rango_fin = body.get("rango_fin")
            if rango_inicio is not None:
                rango_inicio = float(rango_inicio)
            if rango_fin is not None:
                rango_fin = float(rango_fin)
            intervalos = [(float(a), float(b)) for a, b in body.get("intervalos", [])]
            activos = body.get("activos", [])

            with JOBS_LOCK:
                sesion = SESIONES.get(sid)
            if not sesion:
                return self._send_text(404, "Sesión no encontrada")

            try:
                if formato == "zip":
                    nombre_out = nombre_unico(SALIDA, sesion["nombre_base"] + "_dataset_voz", ".zip")
                    ruta_out = os.path.join(SALIDA, nombre_out)
                    info = generar_zip_dataset_voz(
                        sesion["ruta_mp3"], ruta_out, activos,
                        max_minutos=max_minutos, min_dur_seg=min_dur_seg,
                        rango_inicio=rango_inicio, rango_fin=rango_fin
                    )
                else:
                    sufijos = {
                        "aislar": "_mi_voz",
                        "recortar": "_recortado",
                        "silenciar": "_silenciado",
                        "rango_directo": "_tramo",
                    }
                    sufijo = sufijos.get(modo, "_procesado")
                    ext = ".wav" if formato == "wav" else ".mp3"
                    nombre_out = nombre_unico(SALIDA, sesion["nombre_base"] + sufijo, ext)
                    ruta_out = os.path.join(SALIDA, nombre_out)
                    info = generar_audio_procesado(
                        sesion["ruta_mp3"], ruta_out,
                        intervalos_silenciados=intervalos,
                        intervalos_activos=activos,
                        modo=modo, formato=formato,
                        cal=sesion["cal"], mono=sesion["mono"],
                        max_minutos=max_minutos, min_dur_seg=min_dur_seg,
                        pausa_seg=pausa_seg,
                        rango_inicio=rango_inicio, rango_fin=rango_fin
                    )
            except Exception as e:
                return self._send_text(500, str(e))

            return self._send_json(200, {
                "nombre_archivo": nombre_out,
                "nombre_mp3": nombre_out,
                **info
            })

        self._send_text(404, "No encontrado")

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    cargar_sesiones_disco()
    print(f"Iniciando Estudio de Audio en http://localhost:{PORT}  (Ctrl+C para cerrar)")

    def _sinc_inicio():
        try:
            res = verificar_y_sincronizar_git(accion_automatica=True)
            print(f"[GitHub Sync] {res.get('mensaje', '')}")
        except Exception as e:
            print(f"[GitHub Sync Error] {e}")

    threading.Thread(target=_sinc_inicio, daemon=True).start()

    if not os.environ.get("SIN_NAVEGADOR"):
        threading.Timer(1, lambda: webbrowser.open(f"http://localhost:{PORT}")).start()
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
