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
        for k in range(2, min(7, len(X_anc))):
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
<title>Estudio de Audio, Hablantes y Subtítulos · Paso a Paso</title>
<style>
:root {
  --bg: #f8fafc;
  --card: #ffffff;
  --text: #0f172a;
  --muted: #64748b;
  --border: #e2e8f0;
  --primary: #2563eb;
  --primary-hover: #1d4ed8;
  --primary-light: #eff6ff;
  --secondary: #0f766e;
  --secondary-hover: #115e59;
  --danger: #dc2626;
  --danger-bg: #fef2f2;
  --ok: #16a34a;
  --ok-bg: #f0fdf4;
  --warn: #d97706;
  --warn-bg: #fffbeb;
  --radius: 12px;
}
* { box-sizing: border-box; }
body {
  font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  max-width: 1180px;
  margin: 0 auto;
  padding: 20px 16px 80px;
  background: var(--bg);
  color: var(--text);
  line-height: 1.48;
}
header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  flex-wrap: wrap;
  gap: 12px;
  margin-bottom: 18px;
  padding-bottom: 14px;
  border-bottom: 1px solid var(--border);
}
header h1 {
  font-size: 1.45rem;
  margin: 0;
  font-weight: 800;
  color: #0f172a;
}
header p {
  margin: 4px 0 0;
  color: var(--muted);
  font-size: 0.88rem;
}
#git-bar {
  background: #fff;
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 7px 12px;
  font-size: 0.82rem;
  display: flex;
  align-items: center;
  gap: 10px;
}

/* Navegador Principal de 5 Pestañas */
.pasos-stepper {
  display: grid;
  grid-template-columns: repeat(5, 1fr);
  gap: 8px;
  margin-bottom: 20px;
}
@media (max-width: 860px) {
  .pasos-stepper { grid-template-columns: 1fr 1fr; }
}
@media (max-width: 520px) {
  .pasos-stepper { grid-template-columns: 1fr; }
}
.step-btn {
  background: #ffffff;
  border: 2px solid var(--border);
  border-radius: 12px;
  padding: 12px 14px;
  text-align: left;
  cursor: pointer;
  transition: all 0.15s ease;
  display: flex;
  flex-direction: column;
  gap: 3px;
}
.step-btn:hover {
  border-color: #93c5fd;
  background: #f8fafc;
}
.step-btn.activo {
  border-color: var(--primary);
  background: var(--primary-light);
  box-shadow: 0 2px 8px rgba(37,99,235,0.12);
}
.step-btn .step-num {
  font-size: 0.75rem;
  font-weight: 800;
  text-transform: uppercase;
  color: var(--primary);
  letter-spacing: 0.03em;
}
.step-btn .step-title {
  font-size: 0.96rem;
  font-weight: 700;
  color: #0f172a;
}
.step-btn .step-sub {
  font-size: 0.78rem;
  color: var(--muted);
}

.step-section {
  display: none;
}
.step-section.activo {
  display: block;
}

.card {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 20px;
  margin-bottom: 18px;
  box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}
.card h2 {
  font-size: 1.15rem;
  margin: 0 0 8px;
  color: #0f172a;
}
.card h3 {
  font-size: 1rem;
  margin: 0 0 10px;
  color: #1e293b;
}
.sub-desc {
  font-size: 0.88rem;
  color: var(--muted);
  margin: 0 0 16px;
}

.grid-2 {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 16px;
}
.grid-3 {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  gap: 14px;
}
@media (max-width: 780px) {
  .grid-2 { grid-template-columns: 1fr; }
}

.dropzone {
  border: 2px dashed #94a3b8;
  border-radius: var(--radius);
  padding: 26px 18px;
  text-align: center;
  background: #f8fafc;
  cursor: pointer;
  transition: all 0.15s;
}
.dropzone:hover, .dropzone.sobre {
  border-color: var(--primary);
  background: #eff6ff;
}

.campo label {
  display: block;
  font-size: 0.82rem;
  font-weight: 700;
  color: #334155;
  margin-bottom: 4px;
}
input[type="text"], input[type="number"], select, textarea {
  width: 100%;
  font-size: 0.9rem;
  padding: 8px 10px;
  border: 1px solid #cbd5e1;
  border-radius: 8px;
  background: #fff;
  color: var(--text);
  font-family: inherit;
}
button {
  font-family: inherit;
  font-size: 0.9rem;
  font-weight: 600;
  padding: 9px 14px;
  border-radius: 8px;
  border: 1px solid transparent;
  cursor: pointer;
  display: inline-flex;
  align-items: center;
  gap: 6px;
  transition: all 0.15s;
}
button:disabled { opacity: 0.5; cursor: not-allowed; }
.btn-primary { background: var(--primary); color: #fff; }
.btn-primary:hover:not(:disabled) { background: var(--primary-hover); }
.btn-secondary { background: var(--secondary); color: #fff; }
.btn-secondary:hover:not(:disabled) { background: var(--secondary-hover); }
.btn-outline { background: #fff; color: #334155; border-color: #cbd5e1; }
.btn-outline:hover:not(:disabled) { background: #f1f5f9; }
.btn-gold { background: var(--warn-bg); color: #b45309; border-color: #fde68a; }
.btn-gold.activo { background: var(--ok); color: #fff; border-color: var(--ok); }
.btn-danger-soft { background: var(--danger-bg); color: var(--danger); border-color: #fecaca; }
.btn-sm { padding: 6px 10px; font-size: 0.82rem; }
.btn-xs { padding: 4px 8px; font-size: 0.76rem; border-radius: 6px; }

.badge {
  font-size: 0.74rem;
  padding: 2px 8px;
  border-radius: 999px;
  background: #e2e8f0;
  color: #334155;
  font-weight: 700;
  white-space: nowrap;
}
.badge-azul { background: #dbeafe; color: #1e40af; }
.badge-verde { background: #dcfce7; color: #15803d; }
.badge-ambar { background: #fef3c7; color: #b45309; }

.cola-lista {
  list-style: none;
  padding: 0;
  margin: 10px 0 0;
  display: flex;
  flex-direction: column;
  gap: 6px;
}
.cola-item {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 8px 12px;
  background: #f8fafc;
  border: 1px solid var(--border);
  border-radius: 8px;
  font-size: 0.88rem;
}

.progreso-item {
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 12px 14px;
  margin-top: 12px;
  background: #ffffff;
}
.barra-fondo {
  height: 9px;
  background: #e2e8f0;
  border-radius: 999px;
  overflow: hidden;
  margin-top: 6px;
}
.barra-relleno {
  height: 100%;
  background: linear-gradient(90deg, #2563eb, #0d9488);
  width: 0%;
  transition: width 0.25s;
}

/* Diagnóstico claro en Paso 3 */
.diagnostico-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 14px;
  margin-bottom: 16px;
}
@media (max-width: 780px) {
  .diagnostico-grid { grid-template-columns: 1fr; }
}
.diag-box {
  border: 2px solid var(--border);
  border-radius: 10px;
  padding: 14px;
  background: #f8fafc;
}
.diag-box.ok {
  border-color: #86efac;
  background: #f0fdf4;
}
.diag-box.warn {
  border-color: #fde68a;
  background: #fffbeb;
}
.diag-box h4 {
  margin: 0 0 6px;
  font-size: 0.95rem;
}
.diag-box p {
  margin: 0;
  font-size: 0.85rem;
  color: #334155;
}

/* Reproductor sticky Paso 3 */
.player-sticky {
  position: sticky;
  top: 8px;
  z-index: 25;
  background: #0f172a;
  color: #f8fafc;
  padding: 10px 14px;
  border-radius: 10px;
  margin-bottom: 16px;
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 12px;
  box-shadow: 0 6px 16px rgba(0,0,0,0.2);
}
.player-sticky audio {
  flex: 1;
  min-width: 240px;
  height: 36px;
}

/* Tarjetas de hablantes en Paso 3 */
.hablantes-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(310px, 1fr));
  gap: 14px;
  margin-bottom: 18px;
}
.spk-card {
  border: 2px solid var(--border);
  border-radius: 10px;
  padding: 14px;
  background: #fff;
  display: flex;
  flex-direction: column;
  gap: 10px;
}
.spk-card.es-mi-voz {
  border-color: #16a34a;
  background: #f0fdf4;
}
.spk-card.silenciado {
  border-color: #fca5a5;
  background: #fef2f2;
  opacity: 0.82;
}
.spk-frases-preview {
  font-size: 0.8rem;
  color: #475569;
  background: #f8fafc;
  border-left: 3px solid #cbd5e1;
  padding: 6px 8px;
  border-radius: 4px;
  font-style: italic;
}

/* Tabla de segmentos */
.segmentos-wrap {
  max-height: 430px;
  overflow-y: auto;
  border: 1px solid var(--border);
  border-radius: 10px;
  background: #f8fafc;
  padding: 8px;
}
.seg-row {
  display: grid;
  grid-template-columns: auto 155px 1fr auto;
  gap: 8px;
  align-items: start;
  padding: 8px 10px;
  border-radius: 8px;
  background: #fff;
  border: 1px solid #e2e8f0;
  margin-bottom: 6px;
}
@media (max-width: 720px) {
  .seg-row { grid-template-columns: 1fr; }
}
.seg-row.seg-silenciado { opacity: 0.45; background: #fef2f2; }
.seg-row.seg-silenciado .seg-texto { text-decoration: line-through; }
.seg-row.fuera-de-rango { opacity: 0.35; background: #f1f5f9; }
.seg-row.marca-inicio { border-left: 5px solid #16a34a; }
.seg-row.marca-fin { border-right: 5px solid #dc2626; }
.seg-time-btn {
  font-family: monospace;
  font-size: 0.78rem;
  padding: 5px 8px;
  background: #f1f5f9;
  color: #334155;
  border: 1px solid #cbd5e1;
  border-radius: 6px;
}

/* Paso 4: Video + Subtítulos superpuestos */
.video-stage {
  position: relative;
  background: #000;
  border-radius: 12px;
  overflow: hidden;
  max-width: 780px;
  margin: 0 auto 14px;
  aspect-ratio: 16 / 9;
  display: flex;
  align-items: center;
  justify-content: center;
}
.video-stage video {
  width: 100%;
  height: 100%;
}
.sub-overlay {
  position: absolute;
  bottom: 48px;
  left: 6%;
  right: 6%;
  text-align: center;
  pointer-events: none;
  z-index: 10;
}
.sub-overlay span {
  display: inline-block;
  background: rgba(0, 0, 0, 0.82);
  color: #fff;
  font-size: 1.08rem;
  font-weight: 600;
  padding: 6px 14px;
  border-radius: 6px;
  line-height: 1.35;
  text-shadow: 0 1px 2px #000;
}
</style>
</head>
<body>

<header>
  <div>
    <h1>🎙️ Estudio de Audio, Hablantes por Grupo, Subtítulos y Estilo ML</h1>
    <p>Recorte por minuto:segundo · Transcripción IA o Tactiq/YouTube · Reconocimiento de Hablantes · Subtítulos para videos sin audio · Guiones a tu Estilo</p>
  </div>
  <div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap">
    <button type="button" class="btn-outline btn-xs" id="btn-toggle-guia" onclick="toggleGuiaBienvenida()" style="background:#eff6ff;border-color:#93c5fd;color:#1d4ed8">
      ❓ ¿Qué hace esta app? (Ocultar / Ver guía)
    </button>
    <div id="git-bar">
      <span>🐙</span>
      <strong id="git-estado-txt">GitHub: Verificando…</strong>
      <button type="button" class="btn-outline btn-xs" id="btn-git-sinc">🔄 Sincronizar</button>
      <button type="button" class="btn-outline btn-xs" id="btn-git-cfg" style="display:none">⚙️ Configurar</button>
    </div>
  </div>
</header>

<!-- RESUMEN DE BIENVENIDA PARA QUIEN INGRESA POR PRIMERA VEZ -->
<div class="card" id="card-bienvenida-guia" style="background:linear-gradient(135deg,#f8fafc 0%,#eff6ff 100%);border:1.5px solid #bfdbfe;padding:16px 18px;margin-bottom:16px">
  <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:12px;flex-wrap:wrap;margin-bottom:10px">
    <div>
      <h2 style="margin:0;font-size:1.05rem;color:#1e3a8a">👋 ¡Bienvenida/o! ¿Qué hace esta aplicación y cómo empezar?</h2>
      <p style="margin:4px 0 0;font-size:0.84rem;color:#334155">
        Esta aplicación procesa <strong>100% en tu PC</strong> tus grabaciones de audio y video. <strong>No es obligatorio empezar por el Paso 1</strong>: hacé clic directo en la pestaña que necesites según lo que quieras hacer hoy:
      </p>
    </div>
    <button type="button" class="btn-outline btn-xs" onclick="toggleGuiaBienvenida()">▲ Ocultar resumen</button>
  </div>

  <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(205px,1fr));gap:10px;font-size:0.81rem">
    <div style="background:#fff;border:1px solid #dbeafe;border-radius:10px;padding:10px;cursor:pointer" onclick="irAPaso(1)">
      <strong style="color:#1d4ed8;display:block;margin-bottom:3px">1️⃣ Video/Audio a MP3 y Recorte</strong>
      Convertí videos o audios a <code>.MP3</code>, o <strong>recortá Videos (<code>.MP4</code>) y Audios</strong> indicando el <strong>minuto y segundo</strong> exacto de inicio y fin.
    </div>
    <div style="background:#fff;border:1px solid #dbeafe;border-radius:10px;padding:10px;cursor:pointer" onclick="irAPaso(2)">
      <strong style="color:#1d4ed8;display:block;margin-bottom:3px">2️⃣ Transcribir o Subir Texto</strong>
      Transcribí un MP3 con IA (Whisper) <strong>o subí una transcripción ya hecha</strong> (detecta sola si viene de <strong>Tactiq, YouTube, Whisper o SRT</strong>).
    </div>
    <div style="background:#fff;border:1px solid #dbeafe;border-radius:10px;padding:10px;cursor:pointer" onclick="irAPaso(3)">
      <strong style="color:#1d4ed8;display:block;margin-bottom:3px">3️⃣ Hablantes, Mi Voz y Recorte</strong>
      Organizá hablantes en <strong>Grupos</strong> (subiendo lista en Excel), etiquetá automático por <strong>presentación de voz o Tactiq</strong>, silenciá hablantes, recortá por <code>mm:ss</code> y aislá <strong>tu voz limpia</strong>.
    </div>
    <div style="background:#fff;border:1px solid #dbeafe;border-radius:10px;padding:10px;cursor:pointer" onclick="irAPaso(4)">
      <strong style="color:#1d4ed8;display:block;margin-bottom:3px">4️⃣ Subtítulos y Video sin Audio</strong>
      ¿Un video de Drive se grabó sin audio pero tenés la transcripción? Sincronizá el texto con el video y descargá los subtítulos <code>.SRT</code> o el <code>.MP4</code> subtitulado.
    </div>
    <div style="background:#fff;border:1px solid #dbeafe;border-radius:10px;padding:10px;cursor:pointer" onclick="irAPaso(5)">
      <strong style="color:#1d4ed8;display:block;margin-bottom:3px">5️⃣ Mi Estilo (ML) y Guiones</strong>
      Entrená un modelo ML con tus transcripciones para que aprenda tus muletillas y forma de explicar, enviale cualquier <strong>guión</strong> y lo reescribe a <strong>tu estilo</strong>.
    </div>
  </div>
  <!-- DIAGRAMA DE FLUJO VERTICAL (DE ARRIBA HACIA ABAJO) -->
  <div style="margin-top:14px;padding-top:12px;border-top:1px dashed #93c5fd">
    <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;margin-bottom:10px">
      <strong style="font-size:0.9rem;color:#1e3a8a">📊 Diagrama de Flujo del Trabajo (De arriba hacia abajo — hacé clic en cualquier bloque para ir directo):</strong>
      <button type="button" class="btn-outline btn-xs" onclick="const el=document.getElementById('wf-diagrama-box'); el.style.display = el.style.display==='none' ? 'flex' : 'none';">
        ↕ Mostrar / Ocultar diagrama de flujo
      </button>
    </div>

    <div id="wf-diagrama-box" style="display:flex;flex-direction:column;align-items:center;gap:4px;max-width:860px;margin:0 auto;font-size:0.81rem">
      <!-- NODO 1: PASO 1 -->
      <div onclick="irAPaso(1)" style="width:100%;max-width:680px;background:#fff;border:2px solid #3b82f6;border-radius:10px;padding:10px 14px;cursor:pointer;box-shadow:0 2px 5px rgba(37,99,235,0.08);text-align:center">
        <div style="font-weight:800;color:#1d4ed8;font-size:0.88rem">📥 PASO 1 · Cargar Video o Audio · Extraer MP3 o Recortar por Minuto y Segundo (mm:ss)</div>
        <div style="color:#475569;margin-top:2px">Subís un video o audio → Extrae el <code>.MP3</code> o recorta el <strong>Video (<code>.MP4</code>) / Audio (<code>.MP3</code>)</strong> diciendo <em>Desde (min:seg)</em> y <em>Hasta (min:seg)</em></div>
      </div>

      <div style="font-size:1.15rem;color:#2563eb;font-weight:900;line-height:1">↓</div>

      <!-- NODO 2: PASO 2 (Bifurcación Opción A / Opción B) -->
      <div onclick="irAPaso(2)" style="width:100%;max-width:760px;background:#fff;border:2px solid #2563eb;border-radius:10px;padding:10px 14px;cursor:pointer;box-shadow:0 2px 5px rgba(37,99,235,0.08)">
        <div style="font-weight:800;color:#1d4ed8;font-size:0.88rem;text-align:center;margin-bottom:6px">📝 PASO 2 · Obtener la Transcripción (Elegís uno de los 2 caminos)</div>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px">
          <div style="background:#eff6ff;border:1px solid #bfdbfe;border-radius:8px;padding:7px 10px;text-align:center">
            <strong style="color:#1e40af;display:block">🗣️ Camino A: Transcribir acá con IA (Whisper)</strong>
            <span>Transcribe el MP3 en tu PC y separa las voces</span>
          </div>
          <div style="background:#eff6ff;border:1px solid #bfdbfe;border-radius:8px;padding:7px 10px;text-align:center">
            <strong style="color:#1e40af;display:block">📄 Camino B: Subir Transcripción ya hecha</strong>
            <span>Detecta sola si es de <strong>Tactiq, YouTube, Whisper o SRT</strong></span>
          </div>
        </div>
      </div>

      <div style="font-size:1.15rem;color:#2563eb;font-weight:900;line-height:1">↓</div>

      <!-- NODO 3: PASO 3 -->
      <div onclick="irAPaso(3)" style="width:100%;max-width:760px;background:#fff;border:2px solid #059669;border-radius:10px;padding:10px 14px;cursor:pointer;box-shadow:0 2px 5px rgba(5,150,105,0.1);text-align:center">
        <div style="font-weight:800;color:#047857;font-size:0.88rem">👥 PASO 3 · Hablantes por Grupo, Auto-Matcheo Inicial, Recorte (mm:ss) y Aislar Mi Voz</div>
        <div style="color:#334155;margin-top:3px">
          • <strong>Grupos de Hablantes</strong> + Lista desde <strong>Excel (<code>.xlsx</code>)</strong> ·
          • <strong>Auto-Matcheo Inicial</strong> (detecta <em>"mi nombre es..."</em> en el audio o sincroniza con <strong>Tactiq</strong>) y guarda las huellas de voz ·<br>
          • <strong>Silenciar hablantes</strong> · <strong>Recortar Voz y Transcripción por minuto:segundo o frase</strong> ·
          • <strong>Aislar solo tu voz limpia</strong> (<code>.MP3</code> / <code>.WAV</code> / <code>Dataset .ZIP</code>)
        </div>
      </div>

      <!-- FLECHAS DE RAMIFICACIÓN HACIA PASO 4 Y PESTAÑA 5 -->
      <div style="display:grid;grid-template-columns:1fr 1fr;width:100%;max-width:760px;text-align:center;font-size:1.1rem;color:#4f46e5;font-weight:900;line-height:1.1">
        <div>↙</div>
        <div>↘</div>
      </div>

      <!-- NODOS 4 Y 5 EN PARALELO AL FINAL DEL FLUJO -->
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;width:100%;max-width:760px">
        <div onclick="irAPaso(4)" style="background:#fff;border:2px solid #7c3aed;border-radius:10px;padding:10px 12px;cursor:pointer;box-shadow:0 2px 5px rgba(124,58,237,0.1);text-align:center">
          <div style="font-weight:800;color:#6d28d9;font-size:0.86rem">🎬 PASO 4 · Subtítulos y Video sin Audio (Drive)</div>
          <div style="color:#475569;margin-top:3px">
            Sincroniza la transcripción con un video grabado sin audio, recorta el video por <code>mm:ss</code> y exporta los subtítulos <strong><code>.SRT / .VTT</code></strong> o el <strong><code>.MP4</code> subtitulado</strong>.
          </div>
        </div>

        <div onclick="irAPaso(5)" style="background:#fff;border:2px solid #4f46e5;border-radius:10px;padding:10px 12px;cursor:pointer;box-shadow:0 2px 5px rgba(79,70,229,0.1);text-align:center">
          <div style="font-weight:800;color:#4338ca;font-size:0.86rem">🧠 PESTAÑA 5 · Modelo ML de Mi Estilo y Guiones</div>
          <div style="color:#475569;margin-top:3px">
            Entrena un modelo ML con tus transcripciones (aprende tus muletillas y forma de hablar) y convierte cualquier <strong>guión neutro a tu estilo</strong>.
            <div style="margin-top:4px;font-size:0.74rem;color:#6b21a8;background:#faf5ff;border:1px dashed #c084fc;border-radius:5px;padding:2px 6px">
              🚀 <em>A futuro: pasar el guión convertido a audio hablado con tu voz clonada</em>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</div>

<!-- BARRA DE 4 PASOS CLAROS -->
<nav class="pasos-stepper" id="stepper-nav">
  <button type="button" class="step-btn activo" data-step="1">
    <span class="step-num">Paso 1</span>
    <span class="step-title">🎵 Video / Audio a MP3</span>
    <span class="step-sub">Extraer o convertir a MP3 con recorte por minuto</span>
  </button>
  <button type="button" class="step-btn" data-step="2">
    <span class="step-num">Paso 2</span>
    <span class="step-title">📝 Transcripción</span>
    <span class="step-sub">Transcribir con IA o subir de Tactiq / YouTube / Whisper</span>
  </button>
  <button type="button" class="step-btn" data-step="3">
    <span class="step-num">Paso 3</span>
    <span class="step-title">👥 Hablantes y Mi Voz</span>
    <span class="step-sub">Etiquetar por Grupo/Excel, auto-matchear voces y aislar mi voz</span>
  </button>
  <button type="button" class="step-btn" data-step="4">
    <span class="step-num">Paso 4</span>
    <span class="step-title">🎬 Subtítulos y Video</span>
    <span class="step-sub">Sincronizar transcripción con video sin audio (Drive)</span>
  </button>
  <button type="button" class="step-btn" data-step="5">
    <span class="step-num">Pestaña 5 · IA</span>
    <span class="step-title">🧠 Mi Estilo (ML) y Guiones</span>
    <span class="step-sub">Entrenar modelo con mis transcripciones y convertir guiones</span>
  </button>
</nav>

<!-- ====================================================================== -->
<!-- PASO 1: CARGAR VIDEO O AUDIO · EXTRAER MP3 O RECORTAR VIDEO (MM:SS)    -->
<!-- ====================================================================== -->
<section class="step-section activo" id="step-1">
  <div class="card">
    <h2>Paso 1: Cargar Video o Audio · Extraer MP3 o Recortar Video por Minuto y Segundo</h2>
    <p class="sub-desc">
      Cargá un <strong>video</strong> (MP4, MKV, MOV, AVI) o un <strong>audio</strong> (WAV, M4A, MP3). Podés indicar el <strong>minuto y segundo de inicio y fin</strong> (<code>mm:ss</code> o <code>hh:mm:ss</code>) para obtener el <strong>Audio MP3 recortado</strong>, el <strong>Video MP4 recortado</strong> o ambos.
    </p>

    <div class="grid-2" style="margin-bottom:16px">
      <div class="dropzone" id="zona-p1">
        <div style="font-size:2rem;margin-bottom:6px">📥</div>
        <div style="font-weight:700;font-size:1rem">Hacé clic o arrastrá acá tu Video o Audio</div>
        <div style="font-size:0.82rem;color:var(--muted);margin-top:4px">Formatos: MP4, MKV, MOV, AVI, WEBM, WAV, M4A, MP3</div>
        <input type="file" id="arch-p1" accept=".mkv,.mp4,.mov,.avi,.webm,.mp3,.wav,.m4a,.ogg,video/*,audio/*" multiple hidden>
      </div>

      <div style="background:#f8fafc;border:1px solid var(--border);border-radius:12px;padding:16px;display:flex;flex-direction:column;justify-content:space-between;gap:12px">
        <div>
          <strong style="font-size:0.95rem;color:#1e3a8a">✂️ Recorte por Minuto y Segundo (Video o Voz/Audio)</strong>
          <div class="grid-2" style="margin-top:10px;gap:10px">
            <div class="campo">
              <label for="p1-desde">⏱️ Desde (minuto:segundo)</label>
              <input id="p1-desde" type="text" placeholder="ej. 02:15 o 00:10:30" style="font-family:monospace;font-weight:700">
            </div>
            <div class="campo">
              <label for="p1-hasta">⏱️ Hasta (minuto:segundo)</label>
              <input id="p1-hasta" type="text" placeholder="ej. 45:20 o 01:05:00" style="font-family:monospace;font-weight:700">
            </div>
            <div class="campo">
              <label for="p1-modo-salida">¿Qué querés obtener?</label>
              <select id="p1-modo-salida" style="font-weight:700">
                <option value="mp3" selected>🎵 Audio MP3 (extraer / recortar voz)</option>
                <option value="video_mp4">🎬 Video MP4 recortado (entre Desde y Hasta)</option>
                <option value="ambos">⚡ Ambos: Audio MP3 + Video MP4 recortado</option>
              </select>
            </div>
            <div class="campo">
              <label for="p1-cal">Calidad MP3</label>
              <select id="p1-cal">
                <option value="128k">128 kbps</option>
                <option value="192k" selected>192 kbps (recomendada)</option>
                <option value="320k">320 kbps (alta calidad)</option>
              </select>
            </div>
          </div>
          <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:10px;margin-top:10px">
            <label style="display:flex;align-items:center;gap:8px;font-size:0.85rem;cursor:pointer">
              <input type="checkbox" id="p1-mono" checked>
              <span>Canal <strong>Mono</strong> en MP3 (recomendado para voz)</span>
            </label>
            <div style="display:flex;align-items:center;gap:6px;font-size:0.82rem">
              <span>Límite MB:</span>
              <input id="p1-mb" type="number" min="1" step="0.5" placeholder="sin límite" style="width:90px;padding:5px 8px">
            </div>
          </div>
        </div>

        <button type="button" class="btn-primary" id="btn-p1-convertir" disabled style="width:100%;justify-content:center">
          🚀 Procesar / Recortar Archivo(s)
        </button>
      </div>
    </div>

    <div id="p1-cola-box" style="display:none">
      <div style="display:flex;justify-content:space-between;align-items:center">
        <strong id="p1-cola-titulo" style="font-size:0.9rem">Archivos seleccionados (0)</strong>
        <button type="button" class="btn-danger-soft btn-xs" id="btn-p1-limpiar">Vaciar lista</button>
      </div>
      <ul class="cola-lista" id="p1-cola-lista"></ul>
    </div>

    <div id="p1-progreso"></div>
  </div>

  <!-- Recortar rápido un archivo que ya está en /audios sin volver a subirlo -->
  <div class="card">
    <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:12px">
      <div style="flex:1;min-width:260px">
        <h3 style="margin:0">📂 Archivos ya guardados en <code>audios/</code> (Recortar por minuto:segundo o ir al Paso 2)</h3>
        <p style="font-size:0.83rem;color:var(--muted);margin:3px 0 8px">
          Podés elegir un audio o video que ya esté en tu carpeta <code>audios/</code> y recortarlo por minuto y segundo (arriba) sin volver a subirlo, o pasar directo al Paso 2.
        </p>
        <div style="display:flex;gap:8px;flex-wrap:wrap;align-items:center">
          <select id="p1-sel-existente" style="max-width:340px"></select>
          <button type="button" class="btn-outline btn-sm" id="btn-p1-recortar-existente">
            ✂️ Recortar este archivo con el Desde / Hasta de arriba
          </button>
        </div>
      </div>
      <button type="button" class="btn-secondary btn-sm" onclick="irAPaso(2)">👉 Ir directo al Paso 2: Transcripción</button>
    </div>
  </div>
</section>

<!-- ====================================================================== -->
<!-- PASO 2: TRANSCRIBIR CON IA O SUBIR TRANSCRIPCIÓN (TACTIQ/YOUTUBE/ETC) -->
<!-- ====================================================================== -->
<section class="step-section" id="step-2">
  <!-- Sesiones ya guardadas para abrir sin esperar -->
  <div class="card" id="card-sesiones-previas">
    <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:10px;margin-bottom:10px">
      <div>
        <h2 style="margin:0">⚡ Transcripciones ya procesadas (Abrir al instante)</h2>
        <p style="font-size:0.84rem;color:var(--muted);margin:2px 0 0">
          Tus transcripciones se guardan automáticamente en disco. Hacé clic en cualquiera para abrirla en el Paso 3 o Paso 4 sin volver a esperar.
        </p>
      </div>
    </div>
    <div id="lista-sesiones-guardadas" style="display:flex;flex-direction:column;gap:8px">
      <span style="font-size:0.85rem;color:var(--muted)">Cargando transcripciones guardadas…</span>
    </div>
  </div>

  <div class="grid-2">
    <!-- OPCIÓN A: TRANSCRIBIR CON WHISPER -->
    <div class="card">
      <h2>Opción A: 🗣️ Transcribir un audio MP3 acá (con Whisper)</h2>
      <p class="sub-desc">Elegí el MP3 que extrajiste en el Paso 1 (o cualquier audio de tu carpeta <code>audios/</code>) para transcribirlo y separar las voces de los hablantes.</p>

      <div class="campo" style="margin-bottom:12px">
        <label for="p2-sel-mp3">1. Elegí el archivo de audio MP3:</label>
        <select id="p2-sel-mp3">
          <option value="">Cargando audios…</option>
        </select>
      </div>

      <div class="grid-2" style="margin-bottom:12px">
        <div class="campo">
          <label for="p2-modelo">2. Modelo de IA:</label>
          <select id="p2-modelo">
            <option value="base">Whisper Base (más rápido)</option>
            <option value="small" selected>Whisper Small (más preciso)</option>
          </select>
        </div>
        <div class="campo">
          <label for="p2-hablantes">3. Cantidad de hablantes a separar:</label>
          <select id="p2-hablantes">
            <option value="0" selected>Automático (detectar entre 2 y 6)</option>
            <option value="1">1 solo hablante</option>
            <option value="2">2 hablantes</option>
            <option value="3">3 hablantes</option>
            <option value="4">4 hablantes</option>
            <option value="5">5 hablantes</option>
            <option value="6">6 hablantes</option>
          </select>
        </div>
      </div>

      <!-- Recorte opcional de audio antes de transcribir -->
      <div style="background:#f8fafc;border:1px solid var(--border);border-radius:8px;padding:10px;margin-bottom:12px">
        <strong style="font-size:0.82rem;display:block;margin-bottom:6px">✂️ Opcional: Transcribir solo un tramo por minuto y segundo</strong>
        <div class="grid-2" style="gap:8px">
          <div class="campo">
            <label for="p2-desde" style="font-size:0.78rem">⏱️ Desde (min:seg)</label>
            <input type="text" id="p2-desde" placeholder="ej. 03:15 (vacío = desde 00:00)" style="font-family:monospace">
          </div>
          <div class="campo">
            <label for="p2-hasta" style="font-size:0.78rem">⏱️ Hasta (min:seg)</label>
            <input type="text" id="p2-hasta" placeholder="ej. 28:40 (vacío = hasta el final)" style="font-family:monospace">
          </div>
        </div>
      </div>

      <button type="button" class="btn-primary" id="btn-p2-transcribir" style="width:100%;justify-content:center">
        🚀 Transcribir Audio y Pasar al Paso 3
      </button>
      <div id="p2-progreso-whisper"></div>
    </div>

    <!-- OPCIÓN B: SUBIR TRANSCRIPCIÓN YA HECHA -->
    <div class="card">
      <h2>Opción B: 📄 Subir una Transcripción ya hecha</h2>
      <p class="sub-desc">
        Si ya tenés la transcripción, subila acá para seguir directo al etiquetado de hablantes o armado de subtítulos.
        <strong>Detecta automáticamente si el formato es de Tactiq, de YouTube, de Whisper o SRT.</strong>
      </p>

      <div class="dropzone" id="zona-p2-trans" style="padding:20px;margin-bottom:12px">
        <div style="font-size:1.7rem;margin-bottom:4px">📄</div>
        <div style="font-weight:700;font-size:0.95rem" id="lbl-arch-trans">Hacé clic o arrastrá tu archivo de transcripción (.TXT, .SRT, .VTT, .DOCX, .PDF)</div>
        <div style="font-size:0.8rem;color:var(--muted);margin-top:3px">Soporta exportaciones de Tactiq, YouTube, Whisper y subtítulos</div>
        <input type="file" id="arch-p2-trans" accept=".txt,.srt,.vtt,.sbv,.docx,.pdf,.json" hidden>
      </div>

      <div class="campo" style="margin-bottom:10px">
        <label for="p2-trans-mp3">Opcional: Vincular con un audio MP3 de <code>audios/</code> (si querés escuchar las voces)</label>
        <select id="p2-trans-mp3">
          <option value="">— Sin audio (solo trabajar con la transcripción y subtítulos) —</option>
        </select>
      </div>

      <!-- Recorte opcional de transcripción al subir -->
      <div style="background:#f8fafc;border:1px solid var(--border);border-radius:8px;padding:10px;margin-bottom:12px">
        <strong style="font-size:0.82rem;display:block;margin-bottom:6px">✂️ Opcional: Recortar la transcripción al subirla por minuto y segundo</strong>
        <div class="grid-2" style="gap:8px;margin-bottom:6px">
          <div class="campo">
            <label for="p2-trans-desde" style="font-size:0.78rem">⏱️ Desde (min:seg)</label>
            <input type="text" id="p2-trans-desde" placeholder="ej. 05:00" style="font-family:monospace">
          </div>
          <div class="campo">
            <label for="p2-trans-hasta" style="font-size:0.78rem">⏱️ Hasta (min:seg)</label>
            <input type="text" id="p2-trans-hasta" placeholder="ej. 40:00" style="font-family:monospace">
          </div>
        </div>
        <label style="display:flex;align-items:center;gap:6px;font-size:0.8rem;cursor:pointer">
          <input type="checkbox" id="p2-trans-reset">
          <span>Reiniciar tiempos a <code>00:00</code> desde el minuto de inicio</span>
        </label>
      </div>

      <button type="button" class="btn-secondary" id="btn-p2-subir-trans" disabled style="width:100%;justify-content:center">
        📂 Cargar Transcripción y Pasar al Paso 3
      </button>
      <div id="p2-res-import"></div>
    </div>
  </div>
</section>

<!-- ====================================================================== -->
<!-- PASO 3: ETIQUETAR HABLANTES POR GRUPO, AUTO-MATCHEO Y AISLAR MI VOZ    -->
<!-- ====================================================================== -->
<section class="step-section" id="step-3">
  <div id="p3-vacio" class="card" style="text-align:center;padding:40px 20px">
    <h3>Todavía no seleccionaste ninguna transcripción activa</h3>
    <p style="color:var(--muted);font-size:0.9rem">Elegí o abrí una transcripción en el <strong>Paso 2</strong> para etiquetar a los hablantes, matchear voces o extraer tu voz limpia.</p>
    <button type="button" class="btn-primary" onclick="irAPaso(2)">👈 Ir al Paso 2 para abrir o crear una transcripción</button>
  </div>

  <div id="p3-contenido" style="display:none">
    <!-- BARRA DE GRUPO DE HABLANTES -->
    <div class="card" style="background:#f0f7ff;border-color:#bfdbfe">
      <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:12px">
        <div>
          <div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap">
            <strong style="font-size:1.02rem;color:#1e3a8a">🗂️ Grupo de Hablantes activo:</strong>
            <select id="p3-sel-grupo" style="width:auto;min-width:200px;font-weight:700"></select>
            <button type="button" class="btn-outline btn-sm" id="btn-nuevo-grupo">➕ Crear nuevo grupo</button>
          </div>
          <div style="font-size:0.8rem;color:#475569;margin-top:4px">
            Los nombres y las huellas de voz quedan guardados dentro de este grupo para reutilizarlos automáticamente en futuras grabaciones (preparado a futuro para organizar por Categoría / Área / Subgrupo).
          </div>
        </div>
        <div style="display:flex;gap:8px;align-items:center">
          <span class="badge badge-azul" id="badge-info-grupo">0 nombres · 0 voces guardadas</span>
        </div>
      </div>
    </div>

    <!-- CUADRO DE DIAGNÓSTICO CLARO: ¿QUÉ DETECTÓ EL SISTEMA? -->
    <div class="diagnostico-grid">
      <div class="diag-box" id="diag-hablantes-box">
        <h4>👥 1. ¿Cuántos hablantes se detectaron?</h4>
        <p id="diag-hablantes-txt">Analizando…</p>
        <div style="display:flex;align-items:center;gap:6px;flex-wrap:wrap;margin-top:10px">
          <span style="font-size:0.8rem;font-weight:700">¿Querés separar en otra cantidad?:</span>
          <button type="button" class="btn-outline btn-xs btn-recluster-k" data-k="2">2 hablantes</button>
          <button type="button" class="btn-outline btn-xs btn-recluster-k" data-k="3">3 hablantes</button>
          <button type="button" class="btn-outline btn-xs btn-recluster-k" data-k="4">4 hablantes</button>
          <button type="button" class="btn-outline btn-xs btn-recluster-k" data-k="5">5 hablantes</button>
          <button type="button" class="btn-outline btn-xs btn-recluster-k" data-k="6">6 hablantes</button>
        </div>
      </div>

      <div class="diag-box" id="diag-mivoz-box">
        <h4>🎙️ 2. ¿Se detectó tu voz o voces guardadas del grupo?</h4>
        <p id="diag-mivoz-txt">Analizando…</p>
      </div>
    </div>

    <!-- REPRODUCTOR STICKY -->
    <div class="player-sticky" id="p3-player-bar">
      <audio id="p3-audio" controls preload="metadata"></audio>
      <div style="font-size:0.84rem;color:#cbd5e1" id="p3-player-info">
        Usá el botón <strong>"▶ Escuchar ejemplo"</strong> en cada tarjeta abajo para oír la voz de cada hablante.
      </div>
    </div>

    <!-- PANEL ESPECIAL: PRIMER MATCHEO DE VOCES (PRESENTACIONES EN AUDIO / EXCEL / TACTIQ) -->
    <div class="card">
      <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:10px;margin-bottom:10px">
        <div>
          <h3 style="margin:0;color:#1e3a8a">🪄 Herramientas para Etiquetar Hablantes (Listado Excel + Auto-Matcheo Inicial)</h3>
          <p style="font-size:0.83rem;color:var(--muted);margin:3px 0 0">
            Cargá el listado de nombres del grupo y usá el matcheo automático cuando los hablantes dicen su nombre en el audio o si tenés la transcripción de Tactiq.
          </p>
        </div>
      </div>

      <div class="grid-3">
        <!-- 1. Subir Excel / Listado de nombres -->
        <div style="background:#f8fafc;border:1px solid var(--border);border-radius:10px;padding:12px">
          <strong style="font-size:0.88rem;display:block;margin-bottom:4px">📋 A. Subir Excel o Lista de Nombres</strong>
          <p style="font-size:0.79rem;color:var(--muted);margin:0 0 8px">
            Subí un Excel (<code>.xlsx</code>), <code>.csv</code> o <code>.txt</code> con los nombres de los hablantes del grupo para elegirlos de una lista desplegable sin escribir a mano.
          </p>
          <div style="display:flex;gap:6px;flex-wrap:wrap">
            <input type="file" id="arch-listado-nombres" accept=".xlsx,.csv,.txt" hidden>
            <button type="button" class="btn-outline btn-sm" onclick="document.getElementById('arch-listado-nombres').click()">📤 Subir Excel / CSV / TXT</button>
            <button type="button" class="btn-outline btn-sm" id="btn-pegar-nombres">✍️ Pegar nombres</button>
          </div>
          <div id="chips-nombres-grupo" style="display:flex;flex-wrap:wrap;gap:4px;margin-top:8px;max-height:90px;overflow-y:auto"></div>
        </div>

        <!-- 2. Auto-detectar por lo que dicen al presentarse -->
        <div style="background:#f8fafc;border:1px solid var(--border);border-radius:10px;padding:12px">
          <strong style="font-size:0.88rem;display:block;margin-bottom:4px">🗣️ B. Auto-Matchear por Presentación en Audio</strong>
          <p style="font-size:0.79rem;color:var(--muted);margin:0 0 8px">
            Ideal para el primer registro: busca cuando cada hablante dice <em>"mi nombre es…"</em>, <em>"yo soy…"</em> o menciona su nombre del listado y lo etiqueta solo.
          </p>
          <button type="button" class="btn-primary btn-sm" id="btn-auto-presentacion" style="width:100%;justify-content:center">
            🪄 Detectar nombres dichos en el audio
          </button>
          <div id="res-auto-presentacion" style="font-size:0.79rem;margin-top:6px"></div>
        </div>

        <!-- 3. Auto-sincronizar y matchear con Tactiq -->
        <div style="background:#f8fafc;border:1px solid var(--border);border-radius:10px;padding:12px">
          <strong style="font-size:0.88rem;display:block;margin-bottom:4px">🔄 C. Sincronizar y Matchear con Tactiq</strong>
          <p style="font-size:0.79rem;color:var(--muted);margin:0 0 8px">
            Subí la transcripción de Tactiq de esta grabación: el sistema <strong>sincroniza solo el desfase de tiempo</strong> con el audio y asigna los nombres de Tactiq a cada voz.
          </p>
          <input type="file" id="arch-tactiq-sync" accept=".txt,.docx,.srt,.vtt,.json" hidden>
          <button type="button" class="btn-secondary btn-sm" onclick="document.getElementById('arch-tactiq-sync').click()" style="width:100%;justify-content:center">
            📄 Subir Tactiq para sincronizar y etiquetar voces
          </button>
          <div id="res-tactiq-sync" style="font-size:0.79rem;margin-top:6px"></div>
        </div>
      </div>
    </div>

    <!-- TARJETAS DE HABLANTES -->
    <div class="card">
      <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:10px;margin-bottom:12px">
        <div>
          <h2 style="margin:0">👥 Hablantes de esta grabación</h2>
          <p style="font-size:0.84rem;color:var(--muted);margin:2px 0 0">
            Escuchá un ejemplo con <strong>▶ Escuchar</strong>, elegí su nombre del listado (o escribilo) y tocá <strong>💾 Guardar en Grupo</strong> para que su voz quede registrada.
          </p>
        </div>
        <div style="display:flex;gap:6px;flex-wrap:wrap">
          <button type="button" class="btn-outline btn-xs" id="btn-p3-silenciar-todos">🔇 Silenciar todos</button>
          <button type="button" class="btn-outline btn-xs" id="btn-p3-activar-todos">🔊 Activar todos</button>
          <button type="button" class="btn-outline btn-xs" id="btn-p3-nuevo-spk">➕ Agregar hablante</button>
        </div>
      </div>

      <div class="hablantes-grid" id="p3-hablantes-grid"></div>
    </div>

    <!-- RECORTE POR MINUTO Y SEGUNDO (VOZ + TRANSCRIPCIÓN) Y EXTRACCIÓN DE AUDIO -->
    <div class="grid-2">
      <!-- Tarjeta Izquierda: Recorte de Tiempo (Minuto y Segundo + Frases) y Recorte de Transcripción -->
      <div class="card">
        <h3 style="color:#1e3a8a;margin-bottom:4px">✂️ 1. Recortar Voz y Transcripción por Minuto y Segundo (o por Frase)</h3>
        <p style="font-size:0.82rem;color:var(--muted);margin-top:0">
          Escribí el <strong>minuto y segundo</strong> de inicio y fin (ej. <code>04:15</code> a <code>32:40</code> o <code>01:02:10</code>). Este rango recorta tanto el <strong>Audio/Voz</strong> como la <strong>Transcripción</strong>.
        </p>

        <!-- Bloque principal: Minuto y Segundo -->
        <div style="background:#f0f7ff;border:1px solid #bfdbfe;border-radius:10px;padding:12px;margin-bottom:12px">
          <strong style="font-size:0.86rem;color:#1e3a8a;display:block;margin-bottom:8px">⏱️ Indicar Minuto y Segundo exacto de recorte:</strong>
          <div class="grid-2" style="gap:10px;margin-bottom:8px">
            <div class="campo">
              <label for="p3-t-ini">Desde (minuto:segundo):</label>
              <div style="display:flex;gap:4px">
                <input type="text" id="p3-t-ini" placeholder="00:00 (ej. 03:25)" style="font-family:monospace;font-weight:700;font-size:0.95rem">
                <button type="button" class="btn-outline btn-xs" id="btn-p3-tomar-ini-audio" title="Tomar el segundo actual del reproductor de audio">📍 Actual</button>
              </div>
            </div>
            <div class="campo">
              <label for="p3-t-fin">Hasta (minuto:segundo):</label>
              <div style="display:flex;gap:4px">
                <input type="text" id="p3-t-fin" placeholder="fin (ej. 28:50)" style="font-family:monospace;font-weight:700;font-size:0.95rem">
                <button type="button" class="btn-outline btn-xs" id="btn-p3-tomar-fin-audio" title="Tomar el segundo actual del reproductor de audio">📍 Actual</button>
              </div>
            </div>
          </div>
          <div style="display:flex;gap:8px;flex-wrap:wrap;align-items:center">
            <button type="button" class="btn-primary btn-xs" id="btn-p3-aplicar-rango">✅ Aplicar minuto y segundo</button>
            <button type="button" class="btn-outline btn-xs" id="btn-p3-limpiar-rango">✕ Quitar recorte (Usar todo)</button>
            <span id="p3-badge-rango" class="badge badge-azul">Tramo activo: Todo el archivo</span>
          </div>
        </div>

        <!-- Opcional: Buscar por Frase de inicio / fin -->
        <div style="background:#f8fafc;border:1px solid var(--border);border-radius:10px;padding:10px;margin-bottom:12px">
          <strong style="font-size:0.82rem;display:block;margin-bottom:6px">🔎 O buscar el minuto y segundo diciendo la Frase de Inicio / Fin:</strong>
          <div class="grid-2" style="gap:8px;margin-bottom:8px">
            <div class="campo">
              <input type="text" id="p3-f-ini" placeholder='⛳ Frase de inicio (ej. "empezamos")' style="font-size:0.83rem;padding:6px 8px">
              <div id="p3-pill-ini"></div>
            </div>
            <div class="campo">
              <input type="text" id="p3-f-fin" placeholder='🏁 Frase de fin (ej. "hasta acá")' style="font-size:0.83rem;padding:6px 8px">
              <div id="p3-pill-fin"></div>
            </div>
          </div>
          <button type="button" class="btn-outline btn-xs" id="btn-p3-buscar-frases">🔍 Buscar frases y autocompletar minuto:segundo</button>
        </div>

        <!-- Acciones de Recorte sobre la Transcripción -->
        <div style="background:#f0fdf4;border:1px solid #bbf7d0;border-radius:10px;padding:12px">
          <strong style="font-size:0.86rem;color:#166534;display:block;margin-bottom:6px">📄 Recortar y Descargar Transcripción en ese Minuto y Segundo:</strong>
          <label style="display:flex;align-items:center;gap:6px;font-size:0.81rem;margin-bottom:6px;cursor:pointer">
            <input type="checkbox" id="p3-chk-reset-reloj" checked>
            <span>Reiniciar reloj a <code>00:00</code> en el minuto "Desde" (ideal si también recortaste el video/audio)</span>
          </label>
          <div style="display:flex;gap:6px;flex-wrap:wrap">
            <button type="button" class="btn-primary btn-xs" onclick="descargarTranscripcionRecortadaPaso3('txt_ts')">⬇ Descargar .TXT recortado (con tiempos)</button>
            <button type="button" class="btn-outline btn-xs" onclick="descargarTranscripcionRecortadaPaso3('txt_limpio')">⬇ Descargar .TXT (solo texto)</button>
            <button type="button" class="btn-outline btn-xs" onclick="descargarTranscripcionRecortadaPaso3('srt')">⬇ Descargar .SRT recortado</button>
            <button type="button" class="btn-danger-soft btn-xs" id="btn-p3-recortar-sesion-perm" title="Borra de esta sesión lo que está fuera del minuto Desde y Hasta">
              ✂️ Dejar en la sesión solo este tramo
            </button>
          </div>
        </div>
      </div>

      <!-- Tarjeta Derecha: Extraer Voz / Audio Procesado (respetando el recorte de minuto y segundo) -->
      <div class="card">
        <h3 style="color:#1e3a8a;margin-bottom:4px">🎙️ 2. Recortar y Extraer Voz / Audio (en el Minuto y Segundo elegido)</h3>
        <p style="font-size:0.82rem;color:var(--muted);margin-top:0">
          Extrae tu voz limpia, silencia hablantes o recorta el audio directo entre el minuto <strong>Desde</strong> y <strong>Hasta</strong> indicados a la izquierda.
        </p>
        <div class="grid-2" style="gap:10px;margin-bottom:10px">
          <div class="campo">
            <label for="p3-modo-audio">Qué hacer con la voz / el audio:</label>
            <select id="p3-modo-audio">
              <option value="aislar" selected>🎯 Aislar SOLO las voces activas (unir limpio para clonar/entrenar voz)</option>
              <option value="rango_directo">📏 Recorte directo de Audio por minuto y segundo (Desde → Hasta)</option>
              <option value="recortar">✂️ Quitar hablantes silenciados (acortar audio en ese tramo)</option>
              <option value="silenciar">🔇 Poner en mudo (volumen 0) a los hablantes silenciados</option>
            </select>
          </div>
          <div class="campo">
            <label for="p3-fmt-audio">Formato de salida:</label>
            <select id="p3-fmt-audio">
              <option value="mp3" selected>MP3</option>
              <option value="wav">WAV 44.1 kHz (alta calidad para entrenar modelo)</option>
              <option value="zip">📦 Dataset ZIP (clips WAV + metadata.csv)</option>
            </select>
          </div>
          <div class="campo">
            <label for="p3-max-min">Duración máxima:</label>
            <select id="p3-max-min">
              <option value="0" selected>Todo el tramo seleccionado</option>
              <option value="2">Los mejores 2 min</option>
              <option value="5">Los mejores 5 min</option>
              <option value="10">Los mejores 10 min</option>
            </select>
          </div>
          <div class="campo">
            <label for="p3-min-seg">Filtro de frases cortas:</label>
            <select id="p3-min-seg">
              <option value="0">Conservar todas</option>
              <option value="1.0" selected>Ignorar &lt; 1 seg ("sí", "claro")</option>
              <option value="2.0">Ignorar &lt; 2 seg</option>
            </select>
          </div>
        </div>
        <div id="p3-resumen-audio" style="font-size:0.83rem;background:#eff6ff;color:#1e40af;padding:8px 10px;border-radius:8px;margin-bottom:10px"></div>
        <div style="display:flex;gap:8px;flex-wrap:wrap">
          <button type="button" class="btn-primary btn-sm" id="btn-p3-export-audio">💾 Generar Voz / Audio Recortado</button>
          <button type="button" class="btn-secondary btn-sm" onclick="irAPaso(4)">👉 Ir al Paso 4: Subtítulos y Recorte de Video</button>
        </div>
        <div id="p3-res-audio"></div>
      </div>
    </div>

    <!-- DETALLE DE LA TRANSCRIPCIÓN LÍNEA POR LÍNEA -->
    <div class="card">
      <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:10px;margin-bottom:10px">
        <div>
          <h3 style="margin:0">📝 Transcripción línea por línea (<span id="p3-cont-seg">0</span> fragmentos)</h3>
          <label style="display:inline-flex;align-items:center;gap:6px;font-size:0.81rem;color:#1e3a8a;margin-top:4px;cursor:pointer">
            <input type="checkbox" id="p3-chk-ocultar-fuera">
            <span>👁️ Mostrar únicamente las líneas dentro del minuto y segundo de recorte (ocultar el resto)</span>
          </label>
        </div>
        <div style="display:flex;gap:8px;flex-wrap:wrap">
          <input type="text" id="p3-buscar" placeholder="🔍 Buscar palabra o frase…" style="width:200px">
          <select id="p3-filtro-spk" style="width:180px">
            <option value="">Todos los hablantes</option>
          </select>
        </div>
      </div>
      <div class="segmentos-wrap" id="p3-seg-wrap"></div>
    </div>
  </div>
</section>

<!-- ====================================================================== -->
<!-- PASO 4: CONVERTIR A SUBTÍTULOS Y SINCRONIZAR CON VIDEO SIN AUDIO       -->
<!-- ====================================================================== -->
<section class="step-section" id="step-4">
  <div id="p4-vacio" class="card" style="text-align:center;padding:40px 20px">
    <h3>Primero elegí o subí una transcripción en el Paso 2</h3>
    <p style="color:var(--muted);font-size:0.9rem">Para generar los subtítulos y sincronizarlos con tu video, primero abrí o subí la transcripción en el Paso 2.</p>
    <button type="button" class="btn-primary" onclick="irAPaso(2)">👈 Ir al Paso 2</button>
  </div>

  <div id="p4-contenido" style="display:none">
    <!-- BARRA DE VINCULACIÓN EXPLÍCITA: ¿QUÉ TRANSCRIPCIÓN VA CON QUÉ VIDEO? -->
    <div class="card" style="background:#f0f7ff;border:1.5px solid #93c5fd;margin-bottom:16px">
      <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:12px">
        <div style="flex:1;min-width:260px">
          <label for="p4-sel-sesion-activa" style="font-size:0.82rem;font-weight:800;color:#1e3a8a;display:block;margin-bottom:4px">
            📄 1. ¿Qué Transcripción estás usando?:
          </label>
          <select id="p4-sel-sesion-activa" style="font-weight:700;background:#fff"></select>
        </div>
        <div style="font-size:1.3rem;color:#2563eb;font-weight:900">⟷</div>
        <div style="flex:1;min-width:260px">
          <label style="font-size:0.82rem;font-weight:800;color:#1e3a8a;display:block;margin-bottom:4px">
            🎬 2. ¿Con qué Video la estás uniendo?:
          </label>
          <div style="display:flex;gap:8px;align-items:center">
            <span id="p4-badge-video-par" class="badge badge-ambar" style="font-size:0.83rem;padding:6px 10px">Ningún video elegido todavía (elegilo abajo a la izquierda)</span>
            <button type="button" class="btn-primary btn-xs" onclick="document.getElementById('arch-p4-video').click()">🎬 Elegir Video</button>
          </div>
        </div>
      </div>
      <div id="p4-match-diagnostico" style="margin-top:8px;font-size:0.82rem;color:#334155"></div>
    </div>

    <div class="card">
      <h2>Paso 4: Convertir Transcripción a Subtítulos y Sincronizar con Video ("Salva-videos sin audio")</h2>
      <p class="sub-desc">
        Ideal cuando grabaste un video (por ejemplo el que está en Google Drive), el micrófono o los hablantes no salieron en el audio, pero sí tenés la transcripción. Acá podés <strong>sincronizar el tiempo del video con la transcripción</strong> para que los subtítulos calcen exacto.
      </p>

      <div class="grid-2">
        <!-- Columna Izquierda: Reproductor de Video con Subtítulos en Vivo -->
        <div>
          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;flex-wrap:wrap;gap:8px">
            <strong style="font-size:0.92rem">1. Cargar el video local para previsualizar y sincronizar:</strong>
            <input type="file" id="arch-p4-video" accept=".mp4,.mkv,.webm,.mov,video/*" hidden>
            <button type="button" class="btn-outline btn-sm" onclick="document.getElementById('arch-p4-video').click()">🎬 Elegir archivo de Video de tu PC</button>
          </div>

          <div class="video-stage">
            <video id="p4-video" controls></video>
            <div class="sub-overlay"><span id="p4-sub-live">Cargá tu video o dale Play para ver los subtítulos sincronizados acá</span></div>
          </div>
          <div style="font-size:0.8rem;color:var(--muted);text-align:center" id="p4-video-nombre">
            (El video se reproduce directo en tu navegador sin gastarte datos ni subirlo a internet)
          </div>
        </div>

        <!-- Columna Derecha: Controles de Sincronización y Descarga de Subtítulos -->
        <div style="display:flex;flex-direction:column;gap:14px">
          <div style="background:#f0f7ff;border:1px solid #bfdbfe;border-radius:10px;padding:14px">
            <strong style="font-size:0.95rem;color:#1e3a8a;display:block;margin-bottom:6px">
              ⏱️ 2. Sincronizar el Video con la Transcripción (Ajustar desfase)
            </strong>
            <p style="font-size:0.82rem;color:#334155;margin:0 0 10px">
              Si el video empezó a grabarse antes o después que la transcripción, elegí una frase de referencia abajo, buscá ese momento en el video y tocá <strong>"📍 Sincronizar con el segundo actual del video"</strong>.
            </p>

            <div class="campo" style="margin-bottom:8px">
              <label for="p4-sel-frase-ref">Frase de referencia de la transcripción:</label>
              <select id="p4-sel-frase-ref"></select>
            </div>

            <div style="display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin-bottom:10px">
              <button type="button" class="btn-primary btn-sm" id="btn-p4-sync-video">
                📍 Sincronizar esta frase con el segundo actual del video
              </button>
            </div>

            <div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap;padding-top:8px;border-top:1px dashed #93c5fd">
              <label style="font-size:0.83rem;font-weight:700">Desfase aplicado (segundos):</label>
              <input type="number" id="p4-offset-seg" step="0.5" value="0" style="width:95px;font-family:monospace;font-weight:700">
              <button type="button" class="btn-outline btn-xs" onclick="ajustarOffsetSub(-1)">-1 seg</button>
              <button type="button" class="btn-outline btn-xs" onclick="ajustarOffsetSub(+1)">+1 seg</button>
              <button type="button" class="btn-outline btn-xs" onclick="ajustarOffsetSub(0, true)">Reset (0s)</button>
            </div>
          </div>

          <!-- Descargar archivos de subtítulos -->
          <div style="background:#f8fafc;border:1px solid var(--border);border-radius:10px;padding:14px">
            <strong style="font-size:0.95rem;display:block;margin-bottom:8px">
              ⬇ 3. Descargar Subtítulos Sincronizados (Para Drive / YouTube / VLC)
            </strong>
            <label style="display:flex;align-items:center;gap:8px;font-size:0.86rem;margin-bottom:6px;cursor:pointer">
              <input type="checkbox" id="p4-inc-nom" checked>
              <span>Incluir el <strong>nombre del hablante</strong> en cada subtítulo (ej. <code>[Nombre]: texto</code>)</span>
            </label>
            <label style="display:flex;align-items:center;gap:8px;font-size:0.86rem;margin-bottom:12px;cursor:pointer">
              <input type="checkbox" id="p4-inc-sil">
              <span>Incluir también fragmentos de hablantes silenciados</span>
            </label>

            <div style="display:flex;gap:8px;flex-wrap:wrap">
              <button type="button" class="btn-primary btn-sm" onclick="descargarSubtitulosPaso4('srt')">
                🎬 Descargar Subtítulos .SRT (Para Google Drive y YouTube)
              </button>
              <button type="button" class="btn-outline btn-sm" onclick="descargarSubtitulosPaso4('vtt')">
                ⬇ Descargar .VTT
              </button>
              <button type="button" class="btn-outline btn-sm" onclick="descargarSubtitulosPaso4('txt_ts')">
                ⬇ Descargar .TXT con tiempos
              </button>
            </div>
          </div>

          <!-- Recortar Video y/o Unir con Subtítulos por Minuto y Segundo -->
          <div style="background:#f8fafc;border:1px solid var(--border);border-radius:10px;padding:14px">
            <strong style="font-size:0.93rem;color:#1e3a8a;display:block;margin-bottom:6px">
              ✂️ 4. Recortar Video (.MP4) por Minuto y Segundo y/o Unirle Subtítulos
            </strong>
            <p style="font-size:0.8rem;color:var(--muted);margin:0 0 8px">
              Podés recortar el video indicando el <strong>minuto y segundo</strong> de inicio y fin (o dejar vacío para usar el video entero), y elegir si querés solo recortar el video o unirle los subtítulos.
            </p>
            <div class="grid-2" style="gap:8px;margin-bottom:10px">
              <div class="campo">
                <label for="p4-t-ini" style="font-size:0.78rem">⏱️ Desde (min:seg)</label>
                <div style="display:flex;gap:4px">
                  <input type="text" id="p4-t-ini" placeholder="ej. 02:10" style="font-family:monospace;font-weight:700">
                  <button type="button" class="btn-outline btn-xs" id="btn-p4-tomar-ini" title="Usar el segundo actual del video">📍 Actual</button>
                </div>
              </div>
              <div class="campo">
                <label for="p4-t-fin" style="font-size:0.78rem">⏱️ Hasta (min:seg)</label>
                <div style="display:flex;gap:4px">
                  <input type="text" id="p4-t-fin" placeholder="ej. 35:00" style="font-family:monospace;font-weight:700">
                  <button type="button" class="btn-outline btn-xs" id="btn-p4-tomar-fin" title="Usar el segundo actual del video">📍 Actual</button>
                </div>
              </div>
            </div>
            <div style="display:flex;gap:8px;flex-wrap:wrap;align-items:center">
              <select id="p4-modo-unir" style="width:auto;flex:1">
                <option value="soft" selected>⚡ Unir Video + Subtítulos internos (Rápido)</option>
                <option value="burn">🔥 Unir Video + Subtítulos quemados en la imagen</option>
                <option value="solo_recortar_video">✂️ Solo Recortar el Video MP4 (sin subtítulos)</option>
              </select>
              <button type="button" class="btn-secondary btn-sm" id="btn-p4-unir-video">
                🎬 Procesar Video (.MP4)
              </button>
            </div>
            <div id="p4-res-unir" style="margin-top:8px;font-size:0.84rem"></div>
          </div>
        </div>
      </div>
    </div>
  </div>
</section>


<!-- ====================================================================== -->
<!-- PESTAÑA 5: ENTRENAR MODELO ML CON MIS TRANSCRIPCIONES Y CONVERTIR GUIÓN -->
<!-- ====================================================================== -->
<section class="step-section" id="step-5">
  <div class="card" style="background:linear-gradient(135deg,#eff6ff 0%,#f5f3ff 100%);border-color:#c7d2fe">
    <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:12px">
      <div>
        <h2 style="margin:0;color:#1e1b4b">🧠 Laboratorio ML: Entrenar Mi Estilo de Habla y Convertir Guiones</h2>
        <p style="font-size:0.86rem;color:#3730a3;margin:4px 0 0">
          Entrená un modelo propio con tus transcripciones (aprende tus muletillas, conectores, ritmo de frase, voseo y vocabulario real) y enviale cualquier guión o texto para que lo reescriba exactamente como hablás vos.
        </p>
      </div>
      <span class="badge badge-azul" id="p5-badge-estado" style="font-size:0.85rem;padding:6px 12px">Cargando estado del modelo…</span>
    </div>
  </div>

  <div class="grid-2">
    <!-- COLUMNA IZQUIERDA: 1. ENTRENAR MODELO Y VER ADN DE TU ESTILO -->
    <div style="display:flex;flex-direction:column;gap:16px">
      <div class="card">
        <h3 style="color:#1e3a8a;margin-bottom:6px">1. 🎓 Entrenar Modelo ML con mis Transcripciones</h3>
        <p style="font-size:0.82rem;color:var(--muted);margin-top:0">
          Seleccioná qué transcripciones procesadas usar (y cuál hablante sos vos) o subí archivos extra de tus grabaciones para que el modelo aprenda tu forma real de expresarte.
        </p>

        <div class="campo" style="margin-bottom:10px">
          <label>A. Elegir sesiones guardadas y qué hablante usar como "Tu Voz":</label>
          <div id="p5-lista-sesiones-train" style="display:flex;flex-direction:column;gap:6px;max-height:170px;overflow-y:auto;padding:8px;background:#f8fafc;border:1px solid var(--border);border-radius:8px">
            <span style="font-size:0.82rem;color:var(--muted)">Cargando sesiones…</span>
          </div>
        </div>

        <div class="campo" style="margin-bottom:12px">
          <label>B. Opcional: Subir archivos extra de transcripciones tuyas (.TXT, .SRT, .DOCX, .PDF):</label>
          <div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap">
            <input type="file" id="arch-p5-extra" accept=".txt,.srt,.vtt,.docx,.pdf" multiple hidden>
            <button type="button" class="btn-outline btn-sm" onclick="document.getElementById('arch-p5-extra').click()">
              📄 Agregar archivos de transcripción al entrenamiento
            </button>
            <span id="p5-lbl-extra" style="font-size:0.8rem;color:var(--muted)">Ningún archivo extra seleccionado</span>
          </div>
        </div>

        <button type="button" class="btn-primary" id="btn-p5-entrenar" style="width:100%;justify-content:center">
          🧠 Entrenar / Actualizar Modelo con mis Transcripciones
        </button>
        <div id="p5-res-entrenar" style="margin-top:8px;font-size:0.84rem"></div>
      </div>

      <!-- Perfil / ADN aprendido por el modelo -->
      <div class="card" id="p5-card-adn">
        <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;margin-bottom:8px">
          <h3 style="margin:0;color:#1e3a8a">📊 ADN de Tu Estilo Aprendido por el Modelo</h3>
          <a href="/api/estilo_dataset_jsonl" download="mi_estilo_finetuning.jsonl" class="btn-outline btn-xs" style="text-decoration:none" title="Descargar pares Guión Neutro -> Tu Estilo en formato JSONL para Fine-Tuning de LLMs">
            📦 Descargar Dataset .JSONL (Fine-Tuning)
          </a>
        </div>

        <div class="grid-3" style="gap:8px;margin-bottom:12px" id="p5-metricas-grid">
          <div style="background:#f8fafc;border:1px solid var(--border);border-radius:8px;padding:8px;text-align:center">
            <div style="font-size:1.15rem;font-weight:800;color:#1e40af" id="p5-m-frases">0</div>
            <div style="font-size:0.74rem;color:var(--muted)">Frases tuyas analizadas</div>
          </div>
          <div style="background:#f8fafc;border:1px solid var(--border);border-radius:8px;padding:8px;text-align:center">
            <div style="font-size:1.15rem;font-weight:800;color:#1e40af" id="p5-m-pal">0</div>
            <div style="font-size:0.74rem;color:var(--muted)">Palabras en tu corpus</div>
          </div>
          <div style="background:#f8fafc;border:1px solid var(--border);border-radius:8px;padding:8px;text-align:center">
            <div style="font-size:1.15rem;font-weight:800;color:#1e40af" id="p5-m-long">0</div>
            <div style="font-size:0.74rem;color:var(--muted)">Palabras / oración promedio</div>
          </div>
        </div>

        <div style="margin-bottom:10px">
          <label style="font-size:0.8rem;font-weight:700;display:block;margin-bottom:4px">
            🗣️ Tus conectores, muletillas y giros más frecuentes detectados:
          </label>
          <div id="p5-chips-marcadores" style="display:flex;flex-wrap:wrap;gap:5px"></div>
        </div>

        <div style="margin-bottom:10px">
          <label style="font-size:0.8rem;font-weight:700;display:block;margin-bottom:4px">
            ✏️ Ajustar las expresiones que querés que el modelo use al convertir tus guiones (separadas por coma):
          </label>
          <div style="display:flex;gap:6px">
            <input type="text" id="p5-inp-muletillas" placeholder="vamos a ver, o sea, por ejemplo, entonces, fíjense que, acá…">
            <button type="button" class="btn-outline btn-xs" id="btn-p5-guardar-muletillas">💾 Guardar</button>
          </div>
        </div>

        <div>
          <label style="font-size:0.8rem;font-weight:700;display:block;margin-bottom:4px">
            🔤 Tus expresiones y trigramas más repetidos al hablar:
          </label>
          <div id="p5-chips-ngramas" style="display:flex;flex-wrap:wrap;gap:5px"></div>
        </div>
      </div>
    </div>

    <!-- COLUMNA DERECHA: 2. ENVIAR UN GUIÓN Y CONVERTIRLO A MI ESTILO -->
    <div style="display:flex;flex-direction:column;gap:16px">
      <div class="card">
        <h3 style="color:#1e3a8a;margin-bottom:6px">2. ✍️ Enviar un Guión y Convertirlo a Mi Estilo</h3>
        <p style="font-size:0.82rem;color:var(--muted);margin-top:0">
          Pegá un guión formal, un resumen, notas o cualquier texto neutro. El modelo lo transformará a tu forma natural de expresarte.
        </p>

        <div class="campo" style="margin-bottom:10px">
          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px">
            <label for="p5-txt-guion">📄 Guión o texto de entrada:</label>
            <button type="button" class="btn-outline btn-xs" id="btn-p5-ejemplo-guion">✨ Cargar guión de prueba</button>
          </div>
          <textarea id="p5-txt-guion" rows="6" placeholder="Pegá acá tu guión. Por ejemplo: 'A continuación veremos cómo definir una función en Python. Es importante destacar que puedes pasarle múltiples parámetros. Observa aquí cómo debemos retornar el resultado...'"></textarea>
        </div>

        <div class="grid-2" style="gap:10px;margin-bottom:10px">
          <div class="campo">
            <label for="p5-intensidad">Intensidad de tu estilo:</label>
            <select id="p5-intensidad">
              <option value="sutil">🌱 Sutil (Voseo y vocabulario propio, pocas muletillas)</option>
              <option value="equilibrado" selected>🎯 Equilibrado (Recomendado — suena natural a vos)</option>
              <option value="conversacional">🎙️ Muy conversacional (Habla espontánea en vivo con aperturas y cierres)</option>
            </select>
          </div>
          <div class="campo">
            <label for="p5-formato-out">Formato de salida:</label>
            <select id="p5-formato-out">
              <option value="fluido" selected>📝 Párrafos fluidos (para leer o grabar de corrido)</option>
              <option value="teleprompter">🎬 Frases cortas tipo Teleprompter / Clonación XTTS</option>
            </select>
          </div>
        </div>

        <label style="display:flex;align-items:center;gap:8px;font-size:0.82rem;margin-bottom:12px;cursor:pointer">
          <input type="checkbox" id="p5-chk-ollama">
          <span>Usar también <strong>Ollama Local</strong> (si tenés Llama3/Mistral corriendo en tu PC en <code>localhost:11434</code>)</span>
        </label>

        <button type="button" class="btn-primary" id="btn-p5-convertir" style="width:100%;justify-content:center;font-size:1rem;padding:12px">
          🪄 Convertir Guión a Mi Estilo de Hablar
        </button>

        <!-- Resultado del Guión Convertido -->
        <div id="p5-box-resultado" style="display:none;margin-top:14px">
          <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;margin-bottom:6px">
            <strong style="color:var(--ok);font-size:0.95rem">✨ Guión adaptado a tu estilo:</strong>
            <span id="p5-info-out" class="badge badge-verde"></span>
          </div>
          <textarea id="p5-txt-salida" rows="7" style="background:#f0fdf4;border:1.5px solid #86efac;font-size:0.95rem;line-height:1.55"></textarea>

          <div style="display:flex;gap:8px;flex-wrap:wrap;margin-top:10px">
            <button type="button" class="btn-primary btn-sm" id="btn-p5-copiar-out">📋 Copiar Guión Convertido</button>
            <button type="button" class="btn-outline btn-sm" id="btn-p5-descargar-out">⬇ Descargar Guión (.TXT)</button>
            <button type="button" class="btn-outline btn-sm" id="btn-p5-copiar-prompt" title="Copia un prompt con todo tu perfil estilométrico y tus ejemplos reales de habla">
              🧬 Copiar Prompt con mi ADN de Estilo
            </button>
          </div>

          <div style="margin-top:12px;background:#f8fafc;border:1px solid var(--border);border-radius:8px;padding:10px">
            <strong style="font-size:0.8rem;color:#334155;display:block;margin-bottom:4px">
              🔍 Fragmentos reales de tus transcripciones que el modelo usó como referencia para este tema (RAG):
            </strong>
            <div id="p5-rag-ejemplos" style="font-size:0.8rem;color:#475569;display:flex;flex-direction:column;gap:6px"></div>
          </div>
        </div>

        <!-- Deseable a futuro registrado en la UI -->
        <div style="margin-top:14px;background:#faf5ff;border:1px dashed #c084fc;border-radius:10px;padding:12px">
          <div style="display:flex;justify-content:space-between;align-items:center;gap:8px;flex-wrap:wrap">
            <strong style="font-size:0.86rem;color:#6b21a8">🚀 Deseable a futuro (Próxima etapa): Clonar mi voz y pasar este guión a audio con mi voz</strong>
            <span class="badge" style="background:#f3e8ff;color:#6b21a8">Roadmap</span>
          </div>
          <p style="font-size:0.79rem;color:#581c87;margin:4px 0 0">
            Conectar el dataset de voz limpia del <strong>Paso 3</strong> con un motor de clonación vocal (XTTS / F5-TTS / RVC) para que el texto ya convertido a tu estilo acá arriba se pueda transformar con 1 clic en un audio <code>.MP3 / .WAV</code> hablado con tu propia voz clonada.
          </p>
        </div>
      </div>
    </div>
  </div>
</section>


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

window.toggleGuiaBienvenida = function() {
  const box = document.getElementById('card-bienvenida-guia');
  if (!box) return;
  const oculto = box.style.display === 'none';
  box.style.display = oculto ? 'block' : 'none';
  try { localStorage.setItem('ocultar_guia_bienvenida', oculto ? '0' : '1'); } catch (e) {}
};
try {
  if (localStorage.getItem('ocultar_guia_bienvenida') === '1') {
    const b = document.getElementById('card-bienvenida-guia');
    if (b) b.style.display = 'none';
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
  if (num === 2) {
    cargarAudiosDisponibles();
    cargarSesionesGuardadas();
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
  for (const f of files) {
    colaP1.push({ id: Math.random().toString(36).slice(2), file: f, nombre: f.name, bytes: f.size });
  }
  renderColaP1();
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
  tit.textContent = `Archivos listos para extraer/convertir a MP3 (${colaP1.length})`;
  lista.innerHTML = colaP1.map(it => `
    <li class="cola-item">
      <span><strong>${escHtml(it.nombre)}</strong> <small style="color:var(--muted)">(${fmtMB(it.bytes)})</small></span>
      <button type="button" class="btn-danger-soft btn-xs" onclick="quitarColaP1('${it.id}')">✕ Quitar</button>
    </li>
  `).join('');
}

window.quitarColaP1 = function(id) {
  colaP1 = colaP1.filter(x => x.id !== id);
  renderColaP1();
};
document.getElementById('btn-p1-limpiar').onclick = () => { colaP1 = []; renderColaP1(); };

btnP1Convertir.onclick = async () => {
  if (!colaP1.length) return;
  const lotes = [...colaP1];
  colaP1 = [];
  renderColaP1();

  await ejecutarLotePaso1(lotes, false);
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
  if (!d && !h) { alert('Escribí el minuto:segundo en "Desde" y/o "Hasta" arriba para recortar.'); return; }
  await ejecutarLotePaso1([{ nombre: nom }], true);
};

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

    if (!soloAudios.length) {
      selA.innerHTML = '<option value="">(Aún no hay MP3 en /audios — creá uno en el Paso 1)</option>';
    } else {
      selA.innerHTML = soloAudios.map(a => `<option value="${escHtml(a.nombre)}" ${a.nombre === seleccionarNombre ? 'selected' : ''}>${escHtml(a.nombre)} (${fmtMB(a.bytes)})</option>`).join('');
    }
    selB.innerHTML = '<option value="">— Sin audio (solo trabajar con la transcripción y subtítulos) —</option>' +
      soloAudios.map(a => `<option value="${escHtml(a.nombre)}">${escHtml(a.nombre)} (${fmtMB(a.bytes)})</option>`).join('');
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
    cont.innerHTML = lista.map(s => `
      <div class="cola-item" style="background:#f0fdf4;border-color:#bbf7d0">
        <div>
          <strong>🎙️ ${escHtml(s.nombre_original)}</strong>
          <span class="badge badge-azul" style="margin-left:6px">${escHtml(s.modelo)}</span>
          <span style="font-size:0.82rem;color:#334155;margin-left:8px">
            Duración: <strong>${fmtHMS(s.duracion)}</strong> · <strong>${s.cantidad_segmentos}</strong> fragmentos · <strong>${s.cantidad_hablantes}</strong> hablantes detectados
          </span>
        </div>
        <div style="display:flex;gap:6px">
          <button type="button" class="btn-primary btn-sm" onclick="abrirSesionGuardada('${escHtml(s.sesion_id)}', 3)">
            👥 Abrir en Paso 3 (Hablantes)
          </button>
          <button type="button" class="btn-secondary btn-sm" onclick="abrirSesionGuardada('${escHtml(s.sesion_id)}', 4)">
            🎬 Abrir en Paso 4 (Subtítulos/Video)
          </button>
        </div>
      </div>
    `).join('');
  } catch (e) {}
}

window.abrirSesionGuardada = async function(sid, pasoDestino = 3) {
  const r = await fetch('/api/sesion?id=' + encodeURIComponent(sid));
  if (!r.ok) { alert('No se pudo cargar la sesión'); return; }
  const d = await r.json();
  activarSesion(d);
  irAPaso(pasoDestino);
};

// Opción A: Transcribir con Whisper
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
    setTimeout(() => irAPaso(3), 600);
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
    document.getElementById('badge-info-grupo').textContent =
      `Grupo "${gObj.nombre}": ${nombresGrupoActual.length} nombres en lista · ${(gObj.perfiles || []).length} voces con huella`;

    const chips = document.getElementById('chips-nombres-grupo');
    if (!nombresGrupoActual.length) {
      chips.innerHTML = '<span style="font-size:0.78rem;color:var(--muted)">Aún no cargaste lista de nombres en este grupo.</span>';
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
  sesionActiva.segmentos = d.segmentos;
  sesionActiva.hablantes = d.hablantes;
  if (!d.hallazgos.length) {
    resBox.innerHTML = `<span style="color:#b45309">No se encontraron frases explícitas tipo "soy [Nombre]" o "mi nombre es [Nombre]". Podés subir el archivo de Tactiq al lado o elegir el nombre en las tarjetas abajo.</span>`;
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
  sesionActiva.segmentos = d.segmentos;
  sesionActiva.hablantes = d.hablantes;
  resBox.innerHTML = `
    <span style="color:var(--ok);font-weight:700">
      ✔ Sincronizado (desfase detectado: ${d.offset_segundos >= 0 ? '+' : ''}${d.offset_segundos}s) ·
      ${Object.keys(d.mapeo_hablantes || {}).length} hablantes etiquetados automáticamente desde Tactiq.
    </span>
  `;
  await cargarGruposYListado();
  renderPaso3();
};

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
  const boxH = document.getElementById('diag-hablantes-box');
  const txtH = document.getElementById('diag-hablantes-txt');
  const boxV = document.getElementById('diag-mivoz-box');
  const txtV = document.getElementById('diag-mivoz-txt');

  const nSpk = st.hablantes.length;
  if (nSpk <= 1) {
    boxH.className = 'diag-box warn';
    txtH.innerHTML = `⚠️ <strong>Hay 1 solo hablante agrupado</strong> (${st.segmentos.length} frases). Si en la grabación hablan varias personas, hacé clic abajo en <strong>"2 hablantes"</strong>, <strong>"3 hablantes"</strong> o <strong>"4 hablantes"</strong> para separar sus voces.`;
  } else {
    boxH.className = 'diag-box ok';
    txtH.innerHTML = `✅ <strong>Se separaron ${nSpk} hablantes distintos</strong> en los ${st.segmentos.length} fragmentos de la grabación. Abajo tenés una tarjeta por cada hablante para escuchar su voz y ponerle su nombre.`;
  }

  const autoReconocidos = st.hablantes.filter(h => h.auto_etiquetado);
  const miHablante = st.hablantes.find(h => h.id === st.miVozId && !h.silenciado);

  if (miHablante) {
    boxV.className = 'diag-box ok';
    txtV.innerHTML = `✅ <strong>Elegiste cuál es tu voz:</strong> Marcaste a <strong>"${escHtml(miHablante.nombre)}"</strong> como tu voz (${miHablante.cantidad} frases · ${fmtHMS(miHablante.duracion)} hablados). Los demás hablantes están silenciados para que puedas extraer solo tu voz limpia.`;
  } else if (autoReconocidos.length > 0) {
    boxV.className = 'diag-box ok';
    txtV.innerHTML = `🧠 <strong>Voces reconocidas automáticamente:</strong> El sistema reconoció a <strong>${autoReconocidos.map(x => escHtml(x.nombre)).join(', ')}</strong> usando las huellas guardadas en el grupo. Si uno sos vos, tocá <strong>"⭐ Esta es mi voz"</strong> en su tarjeta.`;
  } else {
    boxV.className = 'diag-box warn';
    txtV.innerHTML = `ℹ️ <strong>Todavía no sabés cuál tarjeta corresponde a cada persona:</strong> Tocá el botón <strong>"▶ Escuchar ejemplo"</strong> en cada tarjeta abajo. Cuando escuches la tuya, tocá <strong>"⭐ Esta es mi voz"</strong>; y a los demás elegiles el nombre y tocá <strong>"💾 Guardar en Grupo"</strong>.`;
  }
}

function renderHablantesPaso3() {
  const st = sesionActiva;
  recalcularEstadisticasSesion();
  renderDiagnosticoClaridad();

  const grid = document.getElementById('p3-hablantes-grid');
  const selFiltro = document.getElementById('p3-filtro-spk');
  const filtroActual = selFiltro.value;
  selFiltro.innerHTML = '<option value="">Todos los hablantes</option>';
  grid.innerHTML = '';

  for (const h of st.hablantes) {
    const opt = document.createElement('option');
    opt.value = h.id;
    opt.textContent = `${h.nombre} (${h.cantidad} frases)`;
    selFiltro.appendChild(opt);

    const esMiVoz = st.miVozId === h.id && !h.silenciado;
    const card = document.createElement('div');
    card.className = 'spk-card' + (h.silenciado ? ' silenciado' : '') + (esMiVoz ? ' es-mi-voz' : '');

    const opcionesLista = `<option value="">— Elegir nombre del Grupo (${nombresGrupoActual.length}) —</option>` +
      nombresGrupoActual.map(nom => `<option value="${escHtml(nom)}" ${nom === h.nombre ? 'selected' : ''}>${escHtml(nom)}</option>`).join('');

    const ejemplosTexto = h.muestras.slice(0, 2).map(m => `[${fmtHMS(m.inicio)}] "${escHtml(m.texto)}"`).join('<br>');

    card.innerHTML = `
      <div style="display:flex;justify-content:space-between;align-items:center;gap:8px;flex-wrap:wrap">
        <div style="display:flex;align-items:center;gap:8px">
          <span style="width:14px;height:14px;border-radius:50%;background:${h.color};display:inline-block"></span>
          <strong style="font-size:1.02rem">${escHtml(h.nombre)}</strong>
        </div>
        <div style="display:flex;gap:4px;flex-wrap:wrap">
          ${h.auto_etiquetado ? '<span class="badge badge-azul">🧠 Voz reconocida</span>' : '<span class="badge">Sin guardar huella</span>'}
          ${esMiVoz ? '<span class="badge badge-verde">⭐ Es tu voz</span>' : ''}
          <span class="badge ${h.silenciado ? 'badge-ambar' : 'badge-verde'}">${h.silenciado ? '🔇 Silenciado' : '🔊 Se escucha'}</span>
        </div>
      </div>

      <div style="font-size:0.82rem;color:var(--muted)">
        <strong>${h.cantidad}</strong> intervenciones · <strong>${fmtHMS(h.duracion)}</strong> hablados en total
      </div>

      <div class="spk-frases-preview">
        ${ejemplosTexto || 'Sin frases asignadas'}
      </div>

      <!-- Etiquetar por nombre (desde listado o escribiendo) -->
      <div style="background:#f8fafc;border:1px solid var(--border);border-radius:8px;padding:8px">
        <label style="font-size:0.76rem;font-weight:700;color:#334155;display:block;margin-bottom:4px">
          🏷️ Etiquetar nombre de este hablante (elegí de la lista o escribí):
        </label>
        ${nombresGrupoActual.length ? `<select class="sel-nom-lista" style="margin-bottom:6px;font-size:0.84rem">${opcionesLista}</select>` : ''}
        <div style="display:flex;gap:6px">
          <input type="text" class="inp-nom-manual" value="${escHtml(h.nombre)}" placeholder="Escribir nombre del hablante…" style="font-size:0.86rem;padding:6px 8px">
          <button type="button" class="btn-primary btn-xs btn-guardar-hablante" title="Asigna este nombre y guarda su huella de voz en el Grupo activo para reconocerlo en el futuro">
            💾 Guardar en Grupo
          </button>
        </div>
      </div>

      <!-- Botones de escuchar muestra, marcar Mi Voz y Silenciar -->
      <div style="display:flex;flex-wrap:wrap;gap:6px;align-items:center">
        <button type="button" class="btn-outline btn-sm btn-muestra" ${(!st.nombre_mp3 || !h.muestras.length) ? 'disabled' : ''}>
          ▶ Escuchar ejemplo
        </button>
        <button type="button" class="btn-gold btn-sm btn-mi-voz ${esMiVoz ? 'activo' : ''}">
          ${esMiVoz ? '✔ Es mi voz (activa)' : '⭐ Esta es mi voz'}
        </button>
        <button type="button" class="btn-outline btn-sm btn-mute-spk" style="${h.silenciado ? 'background:#dc2626;color:#fff;border-color:#dc2626' : ''}">
          ${h.silenciado ? '🔇 Silenciado (clic para activar)' : '🔇 Silenciar / Excluir'}
        </button>
      </div>
    `;

    const inpNom = card.querySelector('.inp-nom-manual');
    const selLista = card.querySelector('.sel-nom-lista');
    if (selLista) {
      selLista.onchange = () => {
        if (selLista.value) inpNom.value = selLista.value;
      };
    }

    card.querySelector('.btn-guardar-hablante').onclick = async () => {
      const nuevoNom = inpNom.value.trim();
      if (!nuevoNom) return;
      h.nombre = nuevoNom;
      const r = await fetch('/api/guardar_voz_registro', {
        method: 'POST',
        headers: {'Content-Type':'application/json'},
        body: JSON.stringify({
          sesion_id: st.sesion_id,
          spk_id: h.id,
          nombre: nuevoNom,
          color: h.color,
          grupo: grupoActivo
        })
      });
      if (r.ok) {
        h.auto_etiquetado = true;
        await cargarGruposYListado();
        renderHablantesPaso3();
        renderSegmentosPaso3();
      } else {
        alert(await r.text());
      }
    };

    const btnMuestra = card.querySelector('.btn-muestra');
    btnMuestra.onclick = () => {
      if (!h.muestras.length) return;
      const idx = (st.muestraIdx[h.id] || 0) % h.muestras.length;
      st.muestraIdx[h.id] = idx + 1;
      const m = h.muestras[idx];
      btnMuestra.textContent = `⏭ Ejemplo (${idx + 1}/${h.muestras.length})`;
      reproducirFragmentoPaso3(m.inicio, m.fin, `${h.nombre}: "${m.texto}"`);
    };

    card.querySelector('.btn-mi-voz').onclick = () => {
      st.miVozId = h.id;
      for (const otro of st.hablantes) {
        otro.silenciado = (otro.id !== h.id);
      }
      for (const s of st.segmentos) delete s.override_mute;
      renderHablantesPaso3();
      renderSegmentosPaso3();
      actualizarResumenAudioPaso3();
    };

    card.querySelector('.btn-mute-spk').onclick = () => {
      h.silenciado = !h.silenciado;
      if (h.silenciado && st.miVozId === h.id) st.miVozId = null;
      for (const s of st.segmentos) {
        if (s.hablante === h.id) delete s.override_mute;
      }
      renderHablantesPaso3();
      renderSegmentosPaso3();
      actualizarResumenAudioPaso3();
    };

    grid.appendChild(card);
  }

  selFiltro.value = filtroActual;
}

// Botones rápidos para separar en 2, 3, 4, 5, 6 hablantes
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
      sesionActiva.miVozId = null;
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
  if (!sesionActiva || !sesionActiva.nombre_mp3) return;
  sesionActiva.pararEn = fin;
  audioEl.currentTime = Math.max(0, ini);
  audioEl.play();
  document.getElementById('p3-player-info').textContent = `▶ Escuchando: ${etiqueta} (${fmtHMS(ini)} → ${fmtHMS(fin)})`;
}

function esSegSilenciado(st, seg) {
  if (seg.override_mute !== undefined && seg.override_mute !== null) return Boolean(seg.override_mute);
  const h = st.hablantes.find(x => x.id === seg.hablante);
  return Boolean(h && h.silenciado);
}

function estaEnRango(st, seg) {
  const r0 = st.rangoInicio ?? 0;
  const r1 = st.rangoFin ?? st.duracion;
  return seg.fin > r0 && seg.inicio < r1;
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
      <button type="button" class="seg-time-btn">▶ ${fmtHMS(seg.inicio)} - ${fmtHMS(seg.fin)}</button>
      <select style="border-left:4px solid ${spk.color};font-weight:700;font-size:0.82rem">${opts}</select>
      <div class="seg-texto" contenteditable="true" spellcheck="false" style="font-size:0.9rem;padding:4px">${escHtml(seg.texto)}</div>
      <div style="display:flex;gap:4px">
        <button type="button" class="btn-outline btn-xs btn-ini" title="Marcar inicio de recorte acá">⛳ Inicio</button>
        <button type="button" class="btn-outline btn-xs btn-fin" title="Marcar fin de recorte acá">🏁 Fin</button>
        <button type="button" class="btn-outline btn-xs btn-mute">${sil ? '🔇' : '🔊'}</button>
      </div>
    `;

    row.querySelector('.seg-time-btn').onclick = () => reproducirFragmentoPaso3(seg.inicio, seg.fin, spk.nombre);
    const sel = row.querySelector('select');
    sel.onchange = () => {
      seg.hablante = sel.value;
      delete seg.override_mute;
      renderHablantesPaso3(); renderSegmentosPaso3(); actualizarResumenAudioPaso3();
    };
    const txtEl = row.querySelector('.seg-texto');
    txtEl.onblur = () => { seg.texto = txtEl.textContent.trim(); };
    row.querySelector('.btn-ini').onclick = () => {
      st.rangoInicio = seg.inicio;
      document.getElementById('p3-t-ini').value = fmtHMS(seg.inicio);
      renderSegmentosPaso3(); actualizarResumenAudioPaso3();
    };
    row.querySelector('.btn-fin').onclick = () => {
      st.rangoFin = seg.fin;
      document.getElementById('p3-t-fin').value = fmtHMS(seg.fin);
      renderSegmentosPaso3(); actualizarResumenAudioPaso3();
    };
    row.querySelector('.btn-mute').onclick = () => {
      seg.override_mute = !sil;
      renderSegmentosPaso3(); actualizarResumenAudioPaso3();
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
  const a = document.getElementById('p3-audio');
  if (!a) return;
  document.getElementById('p3-t-ini').value = fmtHMS(a.currentTime || 0);
  aplicarInputsRangoPaso3();
};
document.getElementById('btn-p3-tomar-fin-audio').onclick = () => {
  const a = document.getElementById('p3-audio');
  if (!a) return;
  document.getElementById('p3-t-fin').value = fmtHMS(a.currentTime || 0);
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
  const r0 = st.rangoInicio ?? 0;
  const r1 = st.rangoFin ?? st.duracion;
  const resetReloj = document.getElementById('p3-chk-reset-reloj').checked;
  const shift = (resetReloj && r0 > 0) ? r0 : 0;
  const mapaNom = {};
  for (const h of st.hablantes) mapaNom[h.id] = h.nombre;

  const segs = st.segmentos.filter(s => estaEnRango(st, s) && !esSegSilenciado(st, s));
  if (!segs.length) {
    alert('No hay fragmentos activos dentro de ese rango de minutos y segundos.');
    return;
  }

  let contenido = '';
  let ext = '.txt';
  if (formato === 'srt') {
    contenido = segs.map((s, i) => {
      const ini = Math.max(0, Math.max(s.inicio, r0) - shift);
      const fin = Math.max(ini + 0.4, Math.min(s.fin, r1) - shift);
      return `${i + 1}\n${fmtSrt(ini, ',')} --> ${fmtSrt(fin, ',')}\n[${mapaNom[s.hablante] || s.hablante}] ${s.texto}\n`;
    }).join('\n');
    ext = '_recorte.srt';
  } else if (formato === 'txt_limpio') {
    contenido = segs.map(s => `${mapaNom[s.hablante] || s.hablante}: ${s.texto}`).join('\n');
    ext = '_recorte_texto.txt';
  } else {
    contenido = segs.map(s => {
      const ini = Math.max(0, Math.max(s.inicio, r0) - shift);
      const fin = Math.max(ini + 0.2, Math.min(s.fin, r1) - shift);
      return `[${fmtHMS(ini)} - ${fmtHMS(fin)}] ${mapaNom[s.hablante] || s.hablante}: ${s.texto}`;
    }).join('\n');
    ext = '_recorte_tiempos.txt';
  }

  const blob = new Blob([contenido], { type: 'text/plain;charset=utf-8' });
  const a = document.createElement('a');
  const base = st.nombre_original.replace(/\.[^.]+$/, '');
  a.href = URL.createObjectURL(blob);
  a.download = `${base}${ext}`;
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

  const r = await fetch('/api/recortar_transcripcion_sesion', {
    method: 'POST',
    headers: {'Content-Type':'application/json'},
    body: JSON.stringify({
      sesion_id: sesionActiva.sesion_id,
      desde,
      hasta,
      reiniciar_reloj: reset
    })
  });
  if (!r.ok) { alert(await r.text()); return; }
  const d = await r.json();
  sesionActiva.segmentos = d.segmentos;
  sesionActiva.hablantes = d.hablantes;
  sesionActiva.duracion = d.duracion;
  sesionActiva.rangoInicio = null;
  sesionActiva.rangoFin = null;
  document.getElementById('p3-t-ini').value = '';
  document.getElementById('p3-t-fin').value = '';
  renderPaso3();
};

function actualizarResumenAudioPaso3() {
  if (!sesionActiva) return;
  const st = sesionActiva;
  const modo = document.getElementById('p3-modo-audio').value;
  const minSeg = Number(document.getElementById('p3-min-seg').value || 0);
  const maxMin = Number(document.getElementById('p3-max-min').value || 0);
  const r0 = st.rangoInicio ?? 0;
  const r1 = st.rangoFin ?? st.duracion;
  const badgeRango = document.getElementById('p3-badge-rango');
  if (badgeRango) {
    if (st.rangoInicio !== null || st.rangoFin !== null) {
      badgeRango.textContent = `✂️ Recorte activo: ${fmtHMS(r0)} → ${fmtHMS(r1)}`;
      badgeRango.className = 'badge badge-verde';
    } else {
      badgeRango.textContent = `Tramo activo: Todo (${fmtHMS(st.duracion)})`;
      badgeRango.className = 'badge badge-azul';
    }
  }
  const segsR = st.segmentos.filter(s => estaEnRango(st, s));
  let activos = segsR.filter(s => !esSegSilenciado(st, s));
  if (modo === 'aislar' && minSeg > 0) {
    activos = activos.filter(s => (Math.min(s.fin, r1) - Math.max(s.inicio, r0)) >= minSeg);
  }
  let durAct = modo === 'rango_directo'
    ? Math.max(0, r1 - r0)
    : activos.reduce((acc, s) => acc + Math.max(0, Math.min(s.fin, r1) - Math.max(s.inicio, r0)), 0);
  if (modo === 'aislar' && maxMin > 0 && durAct > maxMin * 60) durAct = maxMin * 60;
  document.getElementById('p3-resumen-audio').innerHTML =
    `⏱️ Rango: <strong>${fmtHMS(r0)} → ${fmtHMS(r1)}</strong> · Fragmentos en rango: <strong>${activos.length}</strong> · Duración de salida: <strong>${fmtHMS(durAct)}</strong>`;
}

['p3-modo-audio','p3-fmt-audio','p3-max-min','p3-min-seg'].forEach(id => {
  document.getElementById(id).onchange = () => actualizarResumenAudioPaso3();
});

document.getElementById('btn-p3-export-audio').onclick = async () => {
  if (!sesionActiva || !sesionActiva.nombre_mp3) {
    alert('Esta sesión no tiene un archivo de audio MP3 vinculado.');
    return;
  }
  const st = sesionActiva;
  const btn = document.getElementById('btn-p3-export-audio');
  const resBox = document.getElementById('p3-res-audio');
  btn.disabled = true;
  resBox.innerHTML = '<div style="margin-top:8px;font-size:0.85rem">⏳ Generando audio…</div>';

  try {
    const r = await fetch('/api/exportar_mp3', {
      method: 'POST', headers: {'Content-Type':'application/json'},
      body: JSON.stringify({
        sesion_id: st.sesion_id,
        modo: document.getElementById('p3-modo-audio').value,
        formato: document.getElementById('p3-fmt-audio').value,
        max_minutos: Number(document.getElementById('p3-max-min').value || 0),
        min_dur_seg: Number(document.getElementById('p3-min-seg').value || 0),
        pausa_seg: 0.3,
        rango_inicio: st.rangoInicio,
        rango_fin: st.rangoFin,
        intervalos: st.segmentos.filter(s => esSegSilenciado(st, s)).map(s => [s.inicio, s.fin]),
        activos: st.segmentos.filter(s => !esSegSilenciado(st, s)).map(s => ({inicio:s.inicio, fin:s.fin, texto:s.texto, confianza:s.confianza||0.8}))
      })
    });
    if (!r.ok) throw new Error(await r.text());
    const d = await r.json();
    resBox.innerHTML = `
      <div style="margin-top:10px;padding:10px;border-radius:8px;background:#f0fdf4;border:1px solid #86efac">
        <div style="font-weight:700;color:var(--ok);margin-bottom:6px">✔ Listo: ${escHtml(d.nombre_archivo)} (${fmtHMS(d.duracion_final)})</div>
        <audio controls src="/audios/${encodeURIComponent(d.nombre_archivo)}" style="width:100%;height:34px;margin-bottom:6px"></audio>
        <a class="btn-primary btn-sm" style="text-decoration:none" href="/audios/${encodeURIComponent(d.nombre_archivo)}" download>⬇ Descargar ${escHtml(d.nombre_archivo)}</a>
      </div>
    `;
  } catch (e) {
    resBox.innerHTML = `<div style="margin-top:8px;color:var(--danger);font-weight:700">✘ ${escHtml(e.message)}</div>`;
  } finally {
    btn.disabled = false;
  }
};

// ==============================================================================
// PASO 4: CONVERTIR A SUBTÍTULOS Y SINCRONIZAR CON VIDEO SIN AUDIO (DRIVE)
// ==============================================================================
async function renderPaso4() {
  if (!sesionActiva) {
    document.getElementById('p4-vacio').style.display = 'block';
    document.getElementById('p4-contenido').style.display = 'none';
    return;
  }
  document.getElementById('p4-vacio').style.display = 'none';
  document.getElementById('p4-contenido').style.display = 'block';

  try {
    const r = await fetch('/api/sesiones_guardadas');
    const lista = await r.json();
    const selSes = document.getElementById('p4-sel-sesion-activa');
    if (selSes) {
      selSes.innerHTML = lista.map(s =>
        `<option value="${escHtml(s.sesion_id)}" ${s.sesion_id === sesionActiva.sesion_id ? 'selected' : ''}>📄 ${escHtml(s.nombre_original)} (${fmtHMS(s.duracion)} · ${s.cantidad_segmentos} frases)</option>`
      ).join('');
      selSes.onchange = async () => {
        await abrirSesionGuardada(selSes.value, 4);
      };
    }
  } catch (e) {}

  const selRef = document.getElementById('p4-sel-frase-ref');
  selRef.innerHTML = sesionActiva.segmentos.slice(0, 60).map((s, idx) =>
    `<option value="${s.inicio}">[${fmtHMS(s.inicio)}] ${escHtml(s.texto.slice(0, 75))}</option>`
  ).join('');
  verificarCoincidenciaVideoTranscripcion();
}

function verificarCoincidenciaVideoTranscripcion() {
  const badge = document.getElementById('p4-badge-video-par');
  const diag = document.getElementById('p4-match-diagnostico');
  const v = document.getElementById('p4-video');
  if (!badge || !diag || !sesionActiva) return;

  if (!archivoVideoP4) {
    const baseOrig = sesionActiva.nombre_original.replace(/\.[^.]+$/, '');
    badge.textContent = 'Falta elegir el archivo de video (.mp4 / .mkv)';
    badge.className = 'badge badge-ambar';
    diag.innerHTML = `💡 <strong>Sugerencia:</strong> Esta transcripción proviene de <code>${escHtml(sesionActiva.nombre_original)}</code> (duración: <strong>${fmtHMS(sesionActiva.duracion)}</strong>). Elegí el video correspondiente con el botón <strong>"🎬 Elegir Video"</strong>.`;
    return;
  }

  badge.textContent = `🎬 ${archivoVideoP4.name} (${fmtMB(archivoVideoP4.size)})`;
  badge.className = 'badge badge-verde';

  const durV = v && v.duration && !isNaN(v.duration) ? v.duration : 0;
  const durT = sesionActiva.duracion || 0;
  const baseV = archivoVideoP4.name.replace(/\.[^.]+$/, '').toLowerCase();
  const baseT = sesionActiva.nombre_original.replace(/\.[^.]+$/, '').toLowerCase();
  const mismoNombre = baseV === baseT || baseV.includes(baseT) || baseT.includes(baseV);

  if (durV > 0) {
    const diff = Math.round(Math.abs(durV - durT));
    if (diff <= 15) {
      diag.innerHTML = `✅ <strong>¡Coincidencia exacta!</strong> El video dura <strong>${fmtHMS(durV)}</strong> y la transcripción abarca <strong>${fmtHMS(durT)}</strong>${mismoNombre ? ' (y tienen el mismo nombre base)' : ''}. Están listos para exportar o unir.`;
    } else {
      diag.innerHTML = `ℹ️ <strong>Duración del video: ${fmtHMS(durV)}</strong> vs <strong>Transcripción: ${fmtHMS(durT)}</strong> (diferencia de ${fmtHMS(diff)}). Si el video empezó a grabarse antes o después que la transcripción, usá el botón <strong>"📍 Sincronizar esta frase con el segundo actual del video"</strong> abajo a la derecha.`;
    }
  } else {
    diag.innerHTML = `🎬 Video vinculado: <strong>${escHtml(archivoVideoP4.name)}</strong> con la transcripción <strong>${escHtml(sesionActiva.nombre_original)}</strong>.`;
  }
}

document.getElementById('arch-p4-video').onchange = e => {
  const f = e.target.files[0];
  if (!f) return;
  archivoVideoP4 = f;
  const v = document.getElementById('p4-video');
  v.src = URL.createObjectURL(f);
  v.onloadedmetadata = () => verificarCoincidenciaVideoTranscripcion();
  document.getElementById('p4-video-nombre').textContent = `🎬 Video cargado para previsualización y sincronización: ${f.name} (${fmtMB(f.size)})`;
  verificarCoincidenciaVideoTranscripcion();
};

// Actualizar subtítulo en vivo sobre el video mientras se reproduce
const videoP4 = document.getElementById('p4-video');
videoP4.ontimeupdate = () => {
  if (!sesionActiva) return;
  const t = videoP4.currentTime;
  const offset = Number(document.getElementById('p4-offset-seg').value || 0);
  const incNom = document.getElementById('p4-inc-nom').checked;
  const incSil = document.getElementById('p4-inc-sil').checked;
  const mapaNom = {};
  for (const h of sesionActiva.hablantes) mapaNom[h.id] = h.nombre;

  const actual = sesionActiva.segmentos.find(s => {
    if (!incSil && esSegSilenciado(sesionActiva, s)) return false;
    const ini = s.inicio + offset;
    const fin = s.fin + offset;
    return t >= ini && t <= fin;
  });

  const subEl = document.getElementById('p4-sub-live');
  if (actual) {
    const pref = incNom ? `[${mapaNom[actual.hablante] || actual.hablante}]: ` : '';
    subEl.textContent = pref + actual.texto;
    subEl.style.display = 'inline-block';
  } else {
    subEl.style.display = 'none';
  }
};

document.getElementById('btn-p4-sync-video').onclick = () => {
  if (!sesionActiva) return;
  const tFraseOrig = Number(document.getElementById('p4-sel-frase-ref').value || 0);
  const tVideoActual = videoP4.currentTime || 0;
  const nuevoOffset = Math.round((tVideoActual - tFraseOrig) * 10) / 10;
  document.getElementById('p4-offset-seg').value = nuevoOffset;
  sesionActiva.offsetSub = nuevoOffset;
  alert(`¡Sincronizado! Se aplicó un ajuste de ${nuevoOffset >= 0 ? '+' : ''}${nuevoOffset} segundos a todos los subtítulos para que calcen con este punto del video.`);
};

window.ajustarOffsetSub = function(delta, reset = false) {
  const inp = document.getElementById('p4-offset-seg');
  const val = reset ? 0 : Math.round((Number(inp.value || 0) + delta) * 10) / 10;
  inp.value = val;
  if (sesionActiva) sesionActiva.offsetSub = val;
};

function fmtSrt(seg, sepMs = ',') {
  const msTotal = Math.max(0, Math.round(seg * 1000));
  const h = Math.floor(msTotal / 3600000);
  const m = Math.floor((msTotal % 3600000) / 60000);
  const s = Math.floor((msTotal % 60000) / 1000);
  const ms = msTotal % 1000;
  return `${String(h).padStart(2,'0')}:${String(m).padStart(2,'0')}:${String(s).padStart(2,'0')}${sepMs}${String(ms).padStart(3,'0')}`;
}

window.descargarSubtitulosPaso4 = function(formato) {
  if (!sesionActiva) return;
  const st = sesionActiva;
  const offset = Number(document.getElementById('p4-offset-seg').value || 0);
  const incNom = document.getElementById('p4-inc-nom').checked;
  const incSil = document.getElementById('p4-inc-sil').checked;
  const mapaNom = {};
  for (const h of st.hablantes) mapaNom[h.id] = h.nombre;

  const segs = st.segmentos.filter(s => estaEnRango(st, s) && (incSil || !esSegSilenciado(st, s)));
  let contenido = '';
  let ext = '.srt';

  if (formato === 'srt') {
    contenido = segs.map((s, i) => {
      const ini = Math.max(0, s.inicio + offset);
      const fin = Math.max(ini + 0.5, s.fin + offset);
      const pref = incNom ? `[${mapaNom[s.hablante] || s.hablante}] ` : '';
      return `${i + 1}\n${fmtSrt(ini, ',')} --> ${fmtSrt(fin, ',')}\n${pref}${s.texto}\n`;
    }).join('\n');
    ext = '_sincronizado.srt';
  } else if (formato === 'vtt') {
    contenido = 'WEBVTT\n\n' + segs.map((s) => {
      const ini = Math.max(0, s.inicio + offset);
      const fin = Math.max(ini + 0.5, s.fin + offset);
      const pref = incNom ? `[${mapaNom[s.hablante] || s.hablante}] ` : '';
      return `${fmtSrt(ini, '.')} --> ${fmtSrt(fin, '.')}\n${pref}${s.texto}\n`;
    }).join('\n');
    ext = '_sincronizado.vtt';
  } else {
    contenido = segs.map(s => {
      const ini = Math.max(0, s.inicio + offset);
      const fin = Math.max(ini + 0.5, s.fin + offset);
      const pref = incNom ? `${mapaNom[s.hablante] || s.hablante}: ` : '';
      return `[${fmtHMS(ini)} - ${fmtHMS(fin)}] ${pref}${s.texto}`;
    }).join('\n');
    ext = '_marcas_sincronizadas.txt';
  }

  const blob = new Blob([contenido], { type: 'text/plain;charset=utf-8' });
  const a = document.createElement('a');
  const base = st.nombre_original.replace(/\.[^.]+$/, '');
  a.href = URL.createObjectURL(blob);
  a.download = `${base}${ext}`;
  a.click();
};

document.getElementById('btn-p4-unir-video').onclick = async () => {
  if (!sesionActiva) return;
  if (!archivoVideoP4) {
    alert('Primero elegí el archivo de video con el botón "🎬 Elegir archivo de Video de tu PC" arriba a la izquierda.');
    return;
  }
  const btn = document.getElementById('btn-p4-unir-video');
  const resDiv = document.getElementById('p4-res-unir');
  btn.disabled = true;
  resDiv.innerHTML = '⏳ Uniendo subtítulos al video con ffmpeg…';

  try {
    const q = new URLSearchParams({
      sesion_id: sesionActiva.sesion_id,
      nombre_video: archivoVideoP4.name,
      modo: document.getElementById('p4-modo-unir').value,
      offset: document.getElementById('p4-offset-seg').value || '0',
      inc_nom: document.getElementById('p4-inc-nom').checked ? '1' : '0',
      inc_sil: document.getElementById('p4-inc-sil').checked ? '1' : '0',
      desde: document.getElementById('p4-t-ini').value.trim(),
      hasta: document.getElementById('p4-t-fin').value.trim()
    });
    const r = await fetch('/api/unir_subtitulos_video?' + q.toString(), {
      method: 'POST',
      body: archivoVideoP4
    });
    if (!r.ok) throw new Error(await r.text());
    const d = await r.json();
    resDiv.innerHTML = `
      <div style="padding:10px;border-radius:8px;background:#f0fdf4;border:1px solid #86efac">
        <strong style="color:var(--ok)">✔ Video con subtítulos creado:</strong> ${escHtml(d.nombre_archivo)}
        <div style="margin-top:6px">
          <a class="btn-primary btn-sm" style="text-decoration:none" href="/audios/${encodeURIComponent(d.nombre_archivo)}" download>
            ⬇ Descargar ${escHtml(d.nombre_archivo)}
          </a>
        </div>
      </div>
    `;
  } catch (e) {
    resDiv.innerHTML = `<div style="color:var(--danger);font-weight:700">✘ ${escHtml(e.message)}</div>`;
  } finally {
    btn.disabled = false;
  }
};

document.getElementById('btn-p4-tomar-ini').onclick = () => {
  document.getElementById('p4-t-ini').value = fmtHMS(videoP4.currentTime || 0);
};
document.getElementById('btn-p4-tomar-fin').onclick = () => {
  document.getElementById('p4-t-fin').value = fmtHMS(videoP4.currentTime || 0);
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
      contSes.innerHTML = '<span style="font-size:0.82rem;color:var(--muted)">No hay sesiones guardadas aún. Podés subir archivos .TXT/.SRT abajo con el botón B.</span>';
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
        disp = "attachment" if ext == ".zip" else "inline"
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
                    if f.lower().endswith((".mp3", ".wav", ".mp4", ".mkv", ".mov", ".webm")):
                        rp = os.path.join(SALIDA, f)
                        if os.path.isfile(rp):
                            archivos.append({
                                "nombre": f,
                                "tipo": "video" if f.lower().endswith((".mp4", ".mkv", ".mov", ".webm")) else "audio",
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
            if largo <= 0:
                return self._send_text(400, "Subí el archivo de video primero.")

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
                    if os.path.exists(ruta_video_in):
                        try:
                            os.remove(ruta_video_in)
                        except OSError:
                            pass
                return self._send_json(200, {"ok": True, "nombre_archivo": nombre_out})

            mapa_nom = {h["id"]: h["nombre"] for h in sesion.get("hablantes", [])}
            mapa_sil = {h["id"]: bool(h.get("silenciado")) for h in sesion.get("hablantes", [])}
            lineas_srt = []
            contador_sub = 1
            t0_shift = float(t_desde_v) if t_desde_v is not None and t_desde_v > 0 else 0.0
            for s in sesion.get("segmentos", []):
                sil = s.get("override_mute") if s.get("override_mute") is not None else mapa_sil.get(s.get("hablante"), False)
                if sil and not inc_sil:
                    continue
                ini_abs = max(0.0, float(s["inicio"]) + offset_seg)
                fin_abs = max(ini_abs + 0.5, float(s["fin"]) + offset_seg)
                if t_desde_v is not None and fin_abs <= t_desde_v:
                    continue
                if t_hasta_v is not None and ini_abs >= t_hasta_v:
                    continue
                ini = max(0.0, ini_abs - t0_shift)
                fin = max(ini + 0.5, (min(t_hasta_v, fin_abs) if t_hasta_v is not None else fin_abs) - t0_shift)
                ms_i = int(round(ini * 1000))
                ms_f = int(round(fin * 1000))
                ts_i = f"{ms_i//3600000:02d}:{(ms_i%3600000)//60000:02d}:{(ms_i%60000)//1000:02d},{ms_i%1000:03d}"
                ts_f = f"{ms_f//3600000:02d}:{(ms_f%3600000)//60000:02d}:{(ms_f%60000)//1000:02d},{ms_f%1000:03d}"
                pref = f"[{mapa_nom.get(s.get('hablante'), s.get('hablante'))}] " if inc_nom else ""
                lineas_srt.append(f"{contador_sub}\n{ts_i} --> {ts_f}\n{pref}{s.get('texto', '')}\n")
                contador_sub += 1

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
                    cmd = ["ffmpeg", "-y", "-i", ruta_vid_base, "-i", ruta_srt_tmp, "-c:v", "copy", "-c:a", "copy", "-c:s", "mov_text", ruta_out]
                r_ff = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
                if r_ff.returncode != 0:
                    return self._send_text(500, f"Error de ffmpeg: {r_ff.stderr[-300:]}")
            finally:
                for rp in (ruta_video_in, ruta_vid_rec_tmp, ruta_srt_tmp):
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
