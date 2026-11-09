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


def agrupar_hablantes(segmentos: list[dict], X: np.ndarray, num_hablantes: int = 0) -> tuple[list[dict], list[dict]]:
    n = len(segmentos)
    if n == 0:
        return [], []
    if n == 1 or num_hablantes == 1:
        for idx, s in enumerate(segmentos):
            s["id"] = idx
            s["hablante"] = "SPK_1"
            s["confianza"] = 1.0
            s.setdefault("silenciado", False)
        return segmentos, _resumen_hablantes(segmentos)

    duraciones = np.array([max(0.1, float(s["fin"]) - float(s["inicio"])) for s in segmentos])
    anclas = duraciones >= 0.65
    if np.sum(anclas) < 4:
        anclas = np.ones(n, dtype=bool)
    X_anc = X[anclas]

    if num_hablantes <= 0:
        mejor_k = 1
        mejor_score = 0.135
        mejor_labels_anc = np.ones(len(X_anc), dtype=int)
        Z = linkage(X_anc, method="ward")
        for k in range(2, min(5, len(X_anc))):
            lbl = fcluster(Z, t=k, criterion="maxclust")
            conteos = [np.sum(lbl == c) for c in np.unique(lbl)]
            if len(X_anc) >= 10 and min(conteos) < 2:
                continue
            sc = _silhouette_score_coseno(X_anc, lbl)
            penalizado = sc - 0.015 * (k - 2)
            if penalizado > mejor_score:
                mejor_score = penalizado
                mejor_k = k
                mejor_labels_anc = lbl
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
    dist = cdist(X, centroides, metric="cosine")
    dist = np.nan_to_num(dist, nan=1.0)
    asignados = np.argmin(dist, axis=1)

    for i in range(n):
        if duraciones[i] < 0.85:
            if 0 < i < n - 1 and asignados[i - 1] == asignados[i + 1]:
                asignados[i] = asignados[i - 1]
            elif i > 0 and (float(segmentos[i]["inicio"]) - float(segmentos[i - 1]["fin"])) < 0.4:
                if dist[i, asignados[i - 1]] - dist[i, asignados[i]] < 0.12:
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
        d_propia = float(dist[idx, c_raw])
        s["confianza"] = round(max(0.0, 1.0 - d_propia), 3)
        s.setdefault("silenciado", False)

    # Calcular centroide real promedio por cada SPK_x para auto-etiquetado
    spk_unicos = sorted(list(set(s["hablante"] for s in segmentos)))
    centroides_dict = {}
    for spk_id in spk_unicos:
        idx_spk = [i for i, s in enumerate(segmentos) if s["hablante"] == spk_id]
        if idx_spk:
            centroides_dict[spk_id] = np.mean(X[idx_spk], axis=0)

    if spk_unicos and centroides_dict:
        centroides_spks = np.vstack([centroides_dict[spk] for spk in spk_unicos])
        coincidencias_voces = emparejar_con_voces_registradas(centroides_spks, spk_unicos)
    else:
        coincidencias_voces = {}

    # Si se detectaron coincidencias con el registro de voces, actualizar nombres de hablantes
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
        modelo = params.get("modelo", "small")
        if modelo not in ("tiny", "base", "small", "medium"):
            modelo = "small"
        idioma = params.get("idioma", "es")
        num_hablantes = int(params.get("hablantes", 0) or 0)
        usar_mp3_existente = bool(params.get("usar_mp3_existente", False))

        base = os.path.splitext(nombre_original)[0]

        if usar_mp3_existente and not desde and not hasta and mb <= 0:
            nombre_mp3 = os.path.basename(ruta_entrada)
            ruta_mp3 = ruta_entrada
        else:
            _actualizar_job(job_id, progreso=5, mensaje="Extrayendo audio MP3 con ffmpeg…")
            nombre_mp3 = nombre_unico(SALIDA, base, ".mp3")
            ruta_mp3 = os.path.join(SALIDA, nombre_mp3)
            extraer_mp3_ffmpeg(ruta_entrada, ruta_mp3, cal=cal, desde=desde, hasta=hasta, mb=mb, mono=mono)

        if solo_extraer:
            _actualizar_job(
                job_id,
                estado="completado",
                progreso=100,
                mensaje="MP3 extraído correctamente.",
                resultado={"modo": "solo_mp3", "nombre_mp3": nombre_mp3}
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

HTML = r"""<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Estudio de Voz · Aislar Mi Voz, Recortar por Frases y Transcribir</title>
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
  --accent: #4f46e5;
  --danger: #dc2626;
  --danger-bg: #fef2f2;
  --danger-border: #fecaca;
  --ok: #16a34a;
  --ok-bg: #f0fdf4;
  --ok-border: #bbf7d0;
  --gold: #d97706;
  --gold-bg: #fffbeb;
  --radius: 12px;
  --shadow-sm: 0 1px 3px rgba(0,0,0,0.05);
  --shadow-md: 0 4px 6px -1px rgba(0,0,0,0.07), 0 2px 4px -2px rgba(0,0,0,0.05);
}
* { box-sizing: border-box; }
body {
  font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  max-width: 1140px;
  margin: 0 auto;
  padding: 24px 16px 80px;
  background: var(--bg);
  color: var(--text);
  line-height: 1.5;
}
header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  flex-wrap: wrap;
  gap: 14px;
  margin-bottom: 24px;
  padding-bottom: 16px;
  border-bottom: 1px solid var(--border);
}
header h1 {
  font-size: 1.6rem;
  margin: 0;
  display: flex;
  align-items: center;
  gap: 10px;
  color: #0f172a;
  font-weight: 800;
  letter-spacing: -0.02em;
}
header p {
  margin: 6px 0 0;
  color: var(--muted);
  font-size: 0.94rem;
}
#git-bar {
  background: #fff;
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 8px 12px;
  font-size: 0.84rem;
  display: flex;
  align-items: center;
  gap: 10px;
  box-shadow: var(--shadow-sm);
}
.card {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 22px;
  margin-bottom: 20px;
  box-shadow: var(--shadow-sm);
  transition: border-color 0.15s;
}
.card-header-line {
  display: flex;
  align-items: center;
  justify-content: space-between;
  flex-wrap: wrap;
  gap: 10px;
  margin-bottom: 16px;
}
.card-header-line h2 {
  font-size: 1.18rem;
  margin: 0;
  display: flex;
  align-items: center;
  gap: 8px;
  color: #1e293b;
  font-weight: 700;
}
.card-badge-info {
  font-size: 0.8rem;
  color: var(--muted);
}

/* Paso 1: Carga */
.fuentes-grid {
  display: grid;
  grid-template-columns: 1.45fr 1fr;
  gap: 16px;
}
@media (max-width: 760px) {
  .fuentes-grid { grid-template-columns: 1fr; }
}
#zona {
  border: 2px dashed #94a3b8;
  border-radius: var(--radius);
  padding: 30px 20px;
  text-align: center;
  background: #f8fafc;
  cursor: pointer;
  transition: all 0.15s ease;
}
#zona:hover, #zona.sobre {
  border-color: var(--primary);
  background: #eff6ff;
}
#zona .icono {
  font-size: 2.2rem;
  margin-bottom: 6px;
}
#zona .titulo-z {
  font-weight: 700;
  font-size: 1.02rem;
  color: #1e293b;
}
#zona .sub-z {
  font-size: 0.84rem;
  color: var(--muted);
  margin-top: 6px;
}
.panel-existentes {
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 16px;
  background: #f8fafc;
  display: flex;
  flex-direction: column;
  justify-content: space-between;
  gap: 12px;
}
.panel-existentes label {
  font-weight: 700;
  font-size: 0.9rem;
  color: #1e293b;
}

/* Cola de archivos */
#cola-box {
  margin-top: 18px;
  display: none;
}
.cola-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 10px;
  font-size: 0.92rem;
  font-weight: 700;
  color: #334155;
}
.cola-lista {
  list-style: none;
  padding: 0;
  margin: 0;
  display: flex;
  flex-direction: column;
  gap: 8px;
}
.cola-item {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  padding: 10px 14px;
  background: #ffffff;
  border: 1px solid var(--border);
  border-radius: 8px;
  font-size: 0.92rem;
  box-shadow: 0 1px 2px rgba(0,0,0,0.03);
}
.cola-item .info {
  display: flex;
  align-items: center;
  gap: 10px;
  overflow: hidden;
}
.cola-item .nombre {
  font-weight: 600;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.badge {
  font-size: 0.74rem;
  padding: 2px 8px;
  border-radius: 999px;
  background: #e2e8f0;
  color: #334155;
  font-weight: 700;
  white-space: nowrap;
}
.badge-azul {
  background: #dbeafe;
  color: #1e40af;
}
.badge-verde {
  background: #dcfce7;
  color: #15803d;
}

/* Paso 2: Selección de Modo y Configuración */
.modos-grid {
  display: grid;
  grid-template-columns: 1.4fr 1fr;
  gap: 14px;
  margin-bottom: 16px;
}
@media (max-width: 760px) {
  .modos-grid { grid-template-columns: 1fr; }
}
.modo-card {
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 16px;
  background: #f8fafc;
  display: flex;
  flex-direction: column;
  justify-content: space-between;
  gap: 12px;
  transition: all 0.15s;
}
.modo-card.destacado {
  background: #eff6ff;
  border-color: #bfdbfe;
}
.modo-card h3 {
  margin: 0 0 6px;
  font-size: 1rem;
  color: #1e3a8a;
  display: flex;
  align-items: center;
  gap: 8px;
}
.modo-card p {
  margin: 0;
  font-size: 0.85rem;
  color: #475569;
}

.opciones-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: 14px;
  margin-bottom: 14px;
  background: #ffffff;
  padding: 14px;
  border: 1px solid var(--border);
  border-radius: 10px;
}
.campo label {
  display: block;
  font-size: 0.83rem;
  font-weight: 700;
  color: #334155;
  margin-bottom: 5px;
}
input[type="text"], input[type="number"], select {
  width: 100%;
  font-size: 0.92rem;
  padding: 8px 10px;
  border: 1px solid #cbd5e1;
  border-radius: 8px;
  background: #fff;
  color: var(--text);
  outline: none;
  transition: border-color 0.15s;
}
input[type="text"]:focus, input[type="number"]:focus, select:focus {
  border-color: var(--primary);
  box-shadow: 0 0 0 2px rgba(37,99,235,0.12);
}
.check-inline {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 0.9rem;
  cursor: pointer;
  user-select: none;
}
.check-inline input {
  width: 16px;
  height: 16px;
  cursor: pointer;
}

.frases-init-box {
  background: #f0f7ff;
  border: 1px dashed #93c5fd;
  border-radius: 10px;
  padding: 14px;
  margin-bottom: 14px;
}
.sub-grid-2 {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 12px;
}
@media (max-width: 650px) {
  .sub-grid-2 { grid-template-columns: 1fr; }
}

/* Botones */
button {
  font-family: inherit;
  font-size: 0.92rem;
  font-weight: 600;
  padding: 9px 15px;
  border-radius: 8px;
  border: 1px solid transparent;
  cursor: pointer;
  display: inline-flex;
  align-items: center;
  gap: 7px;
  transition: all 0.15s ease;
}
button:disabled {
  opacity: 0.55;
  cursor: not-allowed;
}
.btn-primary {
  background: var(--primary);
  color: #fff;
}
.btn-primary:hover:not(:disabled) {
  background: var(--primary-hover);
}
.btn-secondary {
  background: var(--secondary);
  color: #fff;
}
.btn-secondary:hover:not(:disabled) {
  background: var(--secondary-hover);
}
.btn-outline {
  background: #fff;
  color: #334155;
  border-color: #cbd5e1;
}
.btn-outline:hover:not(:disabled) {
  background: #f1f5f9;
}
.btn-gold {
  background: var(--gold-bg);
  color: #b45309;
  border-color: #fde68a;
  padding: 6px 12px;
  font-size: 0.82rem;
}
.btn-gold:hover:not(:disabled) {
  background: #fef3c7;
}
.btn-gold.activo {
  background: #16a34a;
  color: #fff;
  border-color: #16a34a;
}
.btn-danger-soft {
  background: var(--danger-bg);
  color: var(--danger);
  border-color: var(--danger-border);
  padding: 6px 10px;
  font-size: 0.82rem;
}
.btn-sm {
  padding: 6px 11px;
  font-size: 0.83rem;
}
.btn-xs {
  padding: 4px 8px;
  font-size: 0.77rem;
  border-radius: 6px;
}

/* Progreso */
.progreso-item {
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 14px;
  margin-bottom: 14px;
  background: #ffffff;
  box-shadow: var(--shadow-sm);
}
.progreso-top {
  display: flex;
  justify-content: space-between;
  font-size: 0.92rem;
  font-weight: 700;
  margin-bottom: 8px;
}
.barra-fondo {
  height: 10px;
  background: #e2e8f0;
  border-radius: 999px;
  overflow: hidden;
}
.barra-relleno {
  height: 100%;
  background: linear-gradient(90deg, #2563eb, #0d9488);
  width: 0%;
  transition: width 0.25s ease;
}

/* ========================================================================== */
/* ESTUDIO INTERACTIVO ORGANIZADO POR PESTAÑAS (TABS) */
/* ========================================================================== */
.estudio-card {
  border: 2px solid #cbd5e1;
  background: #ffffff;
  padding: 20px;
}

/* Reproductor fijo superior */
.player-sticky {
  position: sticky;
  top: 10px;
  z-index: 30;
  background: #0f172a;
  color: #f8fafc;
  padding: 12px 16px;
  border-radius: 12px;
  margin-bottom: 18px;
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 12px;
  box-shadow: 0 8px 20px -3px rgba(0,0,0,0.25);
  border: 1px solid #334155;
}
.player-sticky audio {
  flex: 1;
  min-width: 250px;
  height: 38px;
}
.player-chips {
  display: flex;
  align-items: center;
  gap: 8px;
  flex-wrap: wrap;
}
.player-chip {
  font-size: 0.78rem;
  padding: 3px 8px;
  border-radius: 6px;
  background: #1e293b;
  border: 1px solid #475569;
  color: #cbd5e1;
  white-space: nowrap;
}
.player-chip.destacado {
  background: #065f46;
  border-color: #059669;
  color: #ecfdf5;
  font-weight: 700;
}
.player-info {
  font-size: 0.83rem;
  color: #94a3b8;
  width: 100%;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

/* Navegación por pestañas */
.tabs-nav {
  display: flex;
  gap: 6px;
  background: #f1f5f9;
  padding: 6px;
  border-radius: 10px;
  margin-bottom: 18px;
  border: 1px solid var(--border);
  overflow-x: auto;
}
.tab-btn {
  background: transparent;
  border: none;
  border-radius: 8px;
  padding: 9px 15px;
  font-size: 0.88rem;
  font-weight: 700;
  color: #475569;
  cursor: pointer;
  display: inline-flex;
  align-items: center;
  gap: 8px;
  white-space: nowrap;
  transition: all 0.15s ease;
}
.tab-btn:hover {
  background: #e2e8f0;
  color: #0f172a;
}
.tab-btn.activo {
  background: #ffffff;
  color: var(--primary);
  box-shadow: 0 1px 3px rgba(0,0,0,0.08);
}
.tab-btn .tab-badge {
  font-size: 0.72rem;
  padding: 1px 6px;
  border-radius: 999px;
  background: #e2e8f0;
  color: #334155;
  font-weight: 700;
}
.tab-btn.activo .tab-badge {
  background: #dbeafe;
  color: #1e40af;
}
.tab-pane {
  display: none;
  animation: tabFadeIn 0.15s ease-out;
}
.tab-pane.activo {
  display: block;
}
@keyframes tabFadeIn {
  from { opacity: 0; transform: translateY(4px); }
  to { opacity: 1; transform: translateY(0); }
}

/* Pestaña 1: Hablantes */
.hablantes-intro-banner {
  background: #f0fdf4;
  border: 1px solid #bbf7d0;
  border-radius: 10px;
  padding: 12px 14px;
  margin-bottom: 16px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  flex-wrap: wrap;
  gap: 10px;
}
.hablantes-intro-banner p {
  margin: 0;
  font-size: 0.88rem;
  color: #166534;
}
.hablantes-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
  gap: 14px;
  margin-bottom: 18px;
}
.spk-card {
  border: 2px solid var(--border);
  border-radius: 10px;
  padding: 14px;
  background: #ffffff;
  transition: all 0.15s ease;
  box-shadow: var(--shadow-sm);
}
.spk-card.es-mi-voz {
  border-color: #16a34a;
  background: #f0fdf4;
}
.spk-card.silenciado {
  background: #fef2f2;
  border-color: #fca5a5;
  opacity: 0.85;
}
.spk-top {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 8px;
}
.spk-dot {
  width: 14px;
  height: 14px;
  border-radius: 50%;
  flex-shrink: 0;
}
.spk-name-input {
  font-weight: 700;
  font-size: 0.96rem;
  padding: 4px 8px !important;
  border: 1px solid transparent !important;
  background: transparent !important;
  border-radius: 6px;
}
.spk-name-input:hover, .spk-name-input:focus {
  border-color: #cbd5e1 !important;
  background: #fff !important;
}
.spk-meta {
  font-size: 0.82rem;
  color: var(--muted);
  margin-bottom: 12px;
}
.spk-actions {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  align-items: center;
}
.mute-toggle {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  font-size: 0.8rem;
  font-weight: 600;
  padding: 5px 9px;
  border-radius: 6px;
  cursor: pointer;
  border: 1px solid #d1d5db;
  background: #f8fafc;
}
.spk-card.silenciado .mute-toggle {
  background: #dc2626;
  color: #fff;
  border-color: #dc2626;
}

/* Memoria de Voces */
.panel-memoria {
  background: #f8fafc;
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 14px;
  margin-bottom: 18px;
}

/* Pestaña 2: Recorte por Frases */
.box-recorte {
  background: #f8fafc;
  border: 1px solid #cbd5e1;
  border-radius: 10px;
  padding: 16px;
  margin-bottom: 16px;
}
.match-pill {
  margin-top: 6px;
  font-size: 0.82rem;
  padding: 6px 10px;
  border-radius: 6px;
  background: #fff;
  border: 1px solid #cbd5e1;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
}

/* Pestaña 3: Transcripción */
.trans-toolbar {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 12px;
  padding-bottom: 10px;
  border-bottom: 1px solid var(--border);
}
.segmentos-wrap {
  max-height: 480px;
  overflow-y: auto;
  border: 1px solid var(--border);
  border-radius: 10px;
  background: #f8fafc;
  padding: 10px;
}
.seg-row {
  display: grid;
  grid-template-columns: auto 150px 1fr auto;
  gap: 10px;
  align-items: start;
  padding: 9px 12px;
  border-radius: 8px;
  background: #fff;
  border: 1px solid #e2e8f0;
  margin-bottom: 6px;
  transition: all 0.15s;
}
@media (max-width: 720px) {
  .seg-row { grid-template-columns: 1fr; }
}
.seg-row.seg-silenciado {
  opacity: 0.45;
  background: #fef2f2;
  border-color: #fecaca;
}
.seg-row.seg-silenciado .seg-texto {
  text-decoration: line-through;
}
.seg-row.fuera-de-rango {
  opacity: 0.35;
  background: #f1f5f9;
}
.seg-row.marca-inicio {
  border-left: 5px solid #16a34a;
}
.seg-row.marca-fin {
  border-right: 5px solid #dc2626;
}
.seg-time-btn {
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 0.78rem;
  padding: 5px 8px;
  background: #f1f5f9;
  color: #334155;
  border: 1px solid #cbd5e1;
  border-radius: 6px;
  white-space: nowrap;
}
.seg-time-btn:hover {
  background: #e2e8f0;
}
.seg-spk-select {
  font-size: 0.82rem !important;
  padding: 5px 6px !important;
  font-weight: 700;
}
.seg-texto {
  font-size: 0.92rem;
  padding: 4px 6px;
  border-radius: 6px;
  border: 1px solid transparent;
  outline: none;
}
.seg-texto:focus {
  border-color: #93c5fd;
  background: #fff;
}

/* Pestaña 4: Exportación */
.export-presets-bar {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 10px;
  margin-bottom: 16px;
}
.preset-card {
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 10px 12px;
  background: #f8fafc;
  cursor: pointer;
  transition: all 0.15s;
}
.preset-card:hover {
  border-color: var(--primary);
  background: #eff6ff;
}
.preset-card strong {
  font-size: 0.88rem;
  display: block;
  color: #1e3a8a;
  margin-bottom: 2px;
}
.preset-card span {
  font-size: 0.78rem;
  color: var(--muted);
}
.export-grid {
  display: grid;
  grid-template-columns: 1.35fr 1fr;
  gap: 16px;
}
@media (max-width: 820px) {
  .export-grid { grid-template-columns: 1fr; }
}
.export-box {
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 16px;
  background: #f8fafc;
}
.export-box h3 {
  font-size: 1.02rem;
  margin: 0 0 12px;
  color: #1e293b;
  display: flex;
  align-items: center;
  gap: 8px;
}
.resultado-mp3-box {
  margin-top: 14px;
  padding: 14px;
  border-radius: 8px;
  background: var(--ok-bg);
  border: 1px solid #86efac;
}
</style>
</head>
<body>

<header>
  <div>
    <h1>🎙️ Estudio de Voz, Recorte y Transcripción</h1>
    <p>Aislar tu voz limpia para clonación (XTTS/RVC) · Recortar por frases de inicio/fin · Diarización y subtítulos</p>
  </div>
  <!-- Barra de estado de GitHub -->
  <div id="git-bar">
    <div style="display:flex;align-items:center;gap:6px">
      <span style="font-size:1.1rem">🐙</span>
      <strong id="git-estado-txt">GitHub: Verificando…</strong>
    </div>
    <div style="display:flex;gap:6px">
      <button type="button" class="btn-outline btn-xs" id="btn-git-sinc" title="Verificar cambios en GitHub o forzar pull/push">🔄 Sincronizar</button>
      <button type="button" class="btn-outline btn-xs" id="btn-git-cfg" style="display:none" title="Conectar repositorio remoto">⚙️ Configurar</button>
    </div>
  </div>
</header>

<!-- PASO 1: SELECCIONAR ARCHIVOS (COLA DE TRABAJO) -->
<div class="card">
  <div class="card-header-line">
    <h2>1. Cargar videos o audios</h2>
    <span class="card-badge-info">No arranca solo: podés cargar varios archivos, configurar y luego iniciar</span>
  </div>
  <div class="fuentes-grid">
    <div id="zona">
      <div class="icono">📥</div>
      <div class="titulo-z">Arrastrá tus archivos acá o hacé clic para explorar</div>
      <div class="sub-z">Soporta MKV, MP4, MOV, AVI, MP3, WAV, M4A</div>
      <input type="file" id="arch" accept=".mkv,.mp4,.mov,.avi,.webm,.mp3,.wav,.m4a,video/*,audio/*" multiple hidden>
    </div>

    <div class="panel-existentes">
      <div>
        <label for="sel-existente">📂 O elegir un audio ya extraído en <code>audios/</code></label>
        <p style="font-size:0.82rem;color:var(--muted);margin:4px 0 8px">Si ya convertiste un video antes, podés abrirlo directo sin volver a subir el archivo pesado.</p>
        <select id="sel-existente">
          <option value="">Cargando lista de audios…</option>
        </select>
      </div>
      <button type="button" class="btn-outline btn-sm" id="btn-agregar-existente" style="align-self:flex-start">➕ Agregar a la cola</button>
    </div>
  </div>

  <div id="cola-box">
    <div class="cola-header">
      <span id="cola-titulo">Archivos listos para procesar (0)</span>
      <button type="button" class="btn-danger-soft" id="btn-limpiar-cola">Vaciar cola</button>
    </div>
    <ul class="cola-lista" id="cola-lista"></ul>
  </div>
</div>

<!-- PASO 2: ELEGIR MODO Y CONFIGURAR -->
<div class="card">
  <div class="card-header-line">
    <h2>2. Configuración y Modo de Ejecución</h2>
    <span id="aviso-cola" style="font-size:0.85rem;color:var(--muted)">Cargá al menos un archivo arriba para activar los botones.</span>
  </div>

  <!-- Tarjetas de Modo -->
  <div class="modos-grid">
    <div class="modo-card destacado">
      <div>
        <h3>🧠 Modo Estudio Completo con IA (Recomendado)</h3>
        <p>Transcribe con Whisper, detecta cada hablante, reconoce tu voz, busca marcas por frases y abre el <strong>Estudio Interactivo</strong> para aislar tu voz limpia o recortar.</p>
      </div>
      <button type="button" class="btn-primary" id="btn-analizar" disabled style="width:100%;justify-content:center">
        🚀 Iniciar Estudio Completo
      </button>
    </div>

    <div class="modo-card">
      <div>
        <h3>⚡ Modo Rápido: Solo Extraer Audio</h3>
        <p>Extrae el audio directo a MP3 o WAV a máxima velocidad sin transcripción ni análisis de hablantes.</p>
      </div>
      <button type="button" class="btn-secondary" id="btn-solo-mp3" disabled style="width:100%;justify-content:center">
        🎵 Extraer Audio Directo
      </button>
    </div>
  </div>

  <!-- Opciones y Ajustes -->
  <div class="opciones-grid">
    <div class="campo">
      <label for="cal">Calidad MP3</label>
      <select id="cal">
        <option value="64k">64 kbps (liviano)</option>
        <option value="128k">128 kbps</option>
        <option value="192k" selected>192 kbps (recomendado)</option>
        <option value="320k">320 kbps (alta fidelidad)</option>
      </select>
    </div>
    <div class="campo">
      <label for="modelo">Modelo de transcripción IA</label>
      <select id="modelo">
        <option value="base">Whisper Base (más rápido)</option>
        <option value="small" selected>Whisper Small (más preciso)</option>
      </select>
    </div>
    <div class="campo">
      <label for="hablantes">Cantidad de hablantes</label>
      <select id="hablantes">
        <option value="0" selected>Automático (detectar)</option>
        <option value="1">1 hablante</option>
        <option value="2">2 hablantes</option>
        <option value="3">3 hablantes</option>
        <option value="4">4 hablantes</option>
        <option value="5">5 hablantes</option>
        <option value="6">6 hablantes</option>
      </select>
    </div>
    <div class="campo">
      <label for="desde">Desde tiempo (opcional)</label>
      <input id="desde" type="text" placeholder="ej. 02:00">
    </div>
    <div class="campo">
      <label for="hasta">Hasta tiempo (opcional)</label>
      <input id="hasta" type="text" placeholder="ej. 15:30">
    </div>
    <div class="campo">
      <label for="mb">Tamaño máx. MB (opcional)</label>
      <input id="mb" type="number" min="1" step="0.5" placeholder="sin límite">
    </div>
  </div>

  <!-- Opción de frases iniciales -->
  <div class="frases-init-box">
    <div style="font-size:0.86rem;font-weight:700;color:#1e3a8a;margin-bottom:8px">
      🔎 Opcional: Detectar marcas de tiempo por frase de inicio y fin desde el inicio
      <span style="font-weight:400;color:var(--muted)">(también disponible en vivo dentro del estudio)</span>
    </div>
    <div class="sub-grid-2">
      <div class="campo">
        <label for="frase-ini-global">⛳ Frase donde empieza el recorte (opcional)</label>
        <input id="frase-ini-global" type="text" placeholder='ej. "bueno empezamos con la clase"'>
      </div>
      <div class="campo">
        <label for="frase-fin-global">🏁 Frase donde termina el recorte (opcional)</label>
        <input id="frase-fin-global" type="text" placeholder='ej. "hasta acá llegamos hoy"'>
      </div>
    </div>
  </div>

  <label class="check-inline">
    <input type="checkbox" id="mono" checked>
    <span>Convertir a <strong>Mono</strong> (recomendado para entrenar modelos de voz: pesa la mitad y evita desfases)</span>
  </label>
</div>

<!-- PROGRESO Y RESULTADOS DIRECTOS -->
<div id="zona-progreso"></div>
<div id="zona-estudios"></div>

<script>
const zona = document.getElementById('zona');
const arch = document.getElementById('arch');
const colaBox = document.getElementById('cola-box');
const colaLista = document.getElementById('cola-lista');
const colaTitulo = document.getElementById('cola-titulo');
const btnAnalizar = document.getElementById('btn-analizar');
const btnSoloMp3 = document.getElementById('btn-solo-mp3');
const avisoCola = document.getElementById('aviso-cola');
const selExistente = document.getElementById('sel-existente');
const zonaProgreso = document.getElementById('zona-progreso');
const zonaEstudios = document.getElementById('zona-estudios');

let cola = [];
const sesiones = {};

// Sincronizador de GitHub en el frontend
async function consultarGitEstado(forzarVerificar = false) {
  try {
    const r = await fetch('/api/git_estado?verificar=' + (forzarVerificar ? '1' : '0'));
    if (!r.ok) return;
    const st = await r.json();
    const txt = document.getElementById('git-estado-txt');
    const btnCfg = document.getElementById('btn-git-cfg');
    if (!txt) return;

    if (!st.tiene_remote) {
      txt.innerHTML = '<span style="color:#d97706">Local (Sin GitHub remoto)</span>';
      btnCfg.style.display = 'inline-block';
      btnCfg.onclick = () => configurarGitRemote();
      return;
    }

    btnCfg.style.display = 'none';
    if (st.estado === 'sincronizado') {
      txt.innerHTML = `<span style="color:#15803d">✔ Al día (${escHtml(st.rama)})</span>`;
    } else if (st.behind > 0) {
      txt.innerHTML = `<span style="color:#2563eb">📥 ${st.behind} cambios en GitHub (Click Sincronizar)</span>`;
    } else if (st.ahead > 0) {
      txt.innerHTML = `<span style="color:#0891b2">📤 ${st.ahead} cambios locales pendientes</span>`;
    } else {
      txt.innerHTML = `<span>${escHtml(st.mensaje || st.estado)}</span>`;
    }
  } catch (e) {
    console.error('Error git estado:', e);
  }
}

async function sincronizarGitManual() {
  const btn = document.getElementById('btn-git-sinc');
  const txt = document.getElementById('git-estado-txt');
  btn.disabled = true;
  txt.textContent = 'Sincronizando con GitHub…';
  try {
    const r = await fetch('/api/git_sincronizar', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ accion: 'auto' })
    });
    if (!r.ok) throw new Error(await r.text());
    const res = await r.json();
    alert(res.mensaje || 'Sincronización completada.');
    consultarGitEstado(false);
  } catch (e) {
    alert('Error al sincronizar con GitHub: ' + e.message);
  } finally {
    btn.disabled = false;
  }
}

async function configurarGitRemote() {
  const url = prompt('Pegá la URL de tu repositorio en GitHub (HTTPS o SSH):\nEjemplo: https://github.com/tu-usuario/extraer-audio.git');
  if (!url || !url.trim()) return;
  try {
    const r = await fetch('/api/git_sincronizar', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ accion: 'configurar_remote', remote_url: url.trim() })
    });
    if (!r.ok) throw new Error(await r.text());
    alert('¡Remoto de GitHub conectado con éxito! Ahora tus cambios se sincronizarán automáticamente.');
    consultarGitEstado(true);
  } catch (e) {
    alert('Error al conectar remoto: ' + e.message);
  }
}

document.getElementById('btn-git-sinc').onclick = () => sincronizarGitManual();
consultarGitEstado(false);

function fmtHMS(seg) {
  if (seg === null || seg === undefined || isNaN(seg)) return '--:--';
  seg = Math.max(0, Number(seg));
  const total = Math.floor(seg);
  const dec = Math.round((seg - total) * 10);
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const s = total % 60;
  const base = h > 0
    ? `${String(h).padStart(2,'0')}:${String(m).padStart(2,'0')}:${String(s).padStart(2,'0')}`
    : `${String(m).padStart(2,'0')}:${String(s).padStart(2,'0')}`;
  return dec > 0 ? `${base}.${dec}` : base;
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

function escHtml(str) {
  return String(str ?? '').replace(/[&<>"']/g, m => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[m]));
}

async function cargarListaAudiosExistentes() {
  try {
    const r = await fetch('/api/audios_existentes');
    const lista = await r.json();
    selExistente.innerHTML = '';
    if (!lista.length) {
      selExistente.innerHTML = '<option value="">(No hay audios previos en /audios)</option>';
      return;
    }
    selExistente.innerHTML = '<option value="">— Elegí un audio de /audios —</option>';
    for (const item of lista) {
      const opt = document.createElement('option');
      opt.value = item.nombre;
      opt.textContent = `${item.nombre} (${fmtMB(item.bytes)})`;
      opt.dataset.bytes = item.bytes;
      selExistente.appendChild(opt);
    }
  } catch (e) {
    selExistente.innerHTML = '<option value="">Error cargando lista</option>';
  }
}
cargarListaAudiosExistentes();

function agregarArchivosACola(files) {
  for (const f of files) {
    cola.push({
      id: Math.random().toString(36).slice(2),
      tipo: 'file',
      file: f,
      nombre: f.name,
      bytes: f.size
    });
  }
  renderCola();
}

document.getElementById('btn-agregar-existente').onclick = () => {
  const val = selExistente.value;
  if (!val) return;
  const opt = selExistente.selectedOptions[0];
  cola.push({
    id: Math.random().toString(36).slice(2),
    tipo: 'existente',
    nombre: val,
    bytes: Number(opt?.dataset?.bytes || 0)
  });
  selExistente.value = '';
  renderCola();
};

document.getElementById('btn-limpiar-cola').onclick = () => {
  cola = [];
  renderCola();
};

function quitarDeCola(id) {
  cola = cola.filter(x => x.id !== id);
  renderCola();
}

function renderCola() {
  if (cola.length === 0) {
    colaBox.style.display = 'none';
    btnAnalizar.disabled = true;
    btnSoloMp3.disabled = true;
    avisoCola.style.display = 'inline';
    return;
  }
  colaBox.style.display = 'block';
  btnAnalizar.disabled = false;
  btnSoloMp3.disabled = false;
  avisoCola.style.display = 'none';
  colaTitulo.textContent = `Archivos en cola listos para procesar (${cola.length})`;
  colaLista.innerHTML = '';

  for (const item of cola) {
    const li = document.createElement('li');
    li.className = 'cola-item';
    const ext = (item.nombre.split('.').pop() || '').toUpperCase();
    const origenBadge = item.tipo === 'existente'
      ? '<span class="badge badge-azul">En /audios</span>'
      : `<span class="badge">${escHtml(ext)}</span>`;
    li.innerHTML = `
      <div class="info">
        ${origenBadge}
        <span class="nombre" title="${escHtml(item.nombre)}">${escHtml(item.nombre)}</span>
        <span style="color:var(--muted);font-size:0.82rem">${fmtMB(item.bytes)}</span>
      </div>
      <button type="button" class="btn-danger-soft" data-id="${item.id}">✕ Quitar</button>
    `;
    li.querySelector('button').onclick = () => quitarDeCola(item.id);
    colaLista.appendChild(li);
  }
}

zona.onclick = () => arch.click();
arch.onchange = () => {
  agregarArchivosACola(arch.files);
  arch.value = '';
};
zona.ondragover = e => { e.preventDefault(); zona.classList.add('sobre'); };
zona.ondragleave = () => zona.classList.remove('sobre');
zona.ondrop = e => {
  e.preventDefault();
  zona.classList.remove('sobre');
  agregarArchivosACola(e.dataTransfer.files);
};

btnAnalizar.onclick = () => procesarCola(false);
btnSoloMp3.onclick = () => procesarCola(true);

async function procesarCola(soloExtraer) {
  if (!cola.length) return;
  const lotes = [...cola];
  cola = [];
  renderCola();

  const paramsBase = {
    cal: document.getElementById('cal').value,
    desde: document.getElementById('desde').value.trim(),
    hasta: document.getElementById('hasta').value.trim(),
    frase_ini: document.getElementById('frase-ini-global').value.trim(),
    frase_fin: document.getElementById('frase-fin-global').value.trim(),
    mb: document.getElementById('mb').value.trim(),
    mono: document.getElementById('mono').checked ? '1' : '0',
    modelo: document.getElementById('modelo').value,
    hablantes: document.getElementById('hablantes').value,
    solo_extraer: soloExtraer ? '1' : '0'
  };

  for (const item of lotes) {
    await ejecutarItem(item, paramsBase);
  }
  cargarListaAudiosExistentes();
}

async function ejecutarItem(item, paramsBase) {
  const pBox = document.createElement('div');
  pBox.className = 'progreso-item';
  pBox.innerHTML = `
    <div class="progreso-top">
      <span>⏳ ${escHtml(item.nombre)} — <span class="msg">Subiendo archivo…</span></span>
      <span class="pct">2%</span>
    </div>
    <div class="barra-fondo"><div class="barra-relleno" style="width:2%"></div></div>
  `;
  zonaProgreso.prepend(pBox);
  const elMsg = pBox.querySelector('.msg');
  const elPct = pBox.querySelector('.pct');
  const elBar = pBox.querySelector('.barra-relleno');

  try {
    const q = new URLSearchParams({
      ...paramsBase,
      nombre: item.nombre,
      usar_mp3_existente: item.tipo === 'existente' ? '1' : '0'
    });

    const r = await fetch('/api/iniciar?' + q.toString(), {
      method: 'POST',
      body: item.tipo === 'file' ? item.file : null
    });
    if (!r.ok) throw new Error(await r.text());
    const { job_id } = await r.json();

    while (true) {
      await new Promise(res => setTimeout(res, 850));
      const st = await fetch('/api/estado?id=' + encodeURIComponent(job_id));
      if (!st.ok) throw new Error(await st.text());
      const d = await st.json();
      elMsg.textContent = d.mensaje || 'Procesando…';
      elPct.textContent = (d.progreso || 0) + '%';
      elBar.style.width = (d.progreso || 0) + '%';

      if (d.estado === 'error') {
        throw new Error(d.error || d.mensaje || 'Error desconocido');
      }
      if (d.estado === 'completado') {
        if (d.resultado.modo === 'solo_mp3') {
          const nom = d.resultado.nombre_mp3;
          pBox.innerHTML = `
            <div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:10px">
              <div><strong style="color:var(--ok)">✔ MP3 extraído:</strong> ${escHtml(nom)}</div>
              <div style="display:flex;gap:10px;align-items:center">
                <audio controls src="/audios/${encodeURIComponent(nom)}" style="height:34px"></audio>
                <a class="btn-primary btn-sm" style="text-decoration:none" href="/audios/${encodeURIComponent(nom)}" download>⬇ Descargar MP3</a>
              </div>
            </div>
          `;
        } else {
          pBox.remove();
          montarEstudio(d.resultado);
        }
        break;
      }
    }
  } catch (e) {
    pBox.style.background = '#fef2f2';
    pBox.style.borderColor = '#fecaca';
    pBox.innerHTML = `<div style="color:var(--danger);font-weight:600">✘ Error con ${escHtml(item.nombre)}: ${escHtml(e.message)}</div>`;
  }
}

// ==============================================================================
// ESTUDIO INTERACTIVO ORGANIZADO EN 4 PESTAÑAS (TABS)
// ==============================================================================

function montarEstudio(datos) {
  const sid = datos.sesion_id;
  const rd = datos.rango_detectado || {};
  sesiones[sid] = {
    ...datos,
    miVozId: null,
    muestraIdx: {},
    pararEn: null,
    filtroTexto: '',
    filtroSpk: '',
    rangoInicio: rd.inicio_seg ?? null,
    rangoFin: rd.fin_seg ?? null,
    matchIni: rd.match_ini ?? null,
    matchFin: rd.match_fin ?? null
  };

  const card = document.createElement('div');
  card.className = 'card estudio-card';
  card.id = 'estudio-' + sid;
  card.innerHTML = `
    <!-- Encabezado del Estudio -->
    <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:10px;margin-bottom:14px">
      <div>
        <h2 style="margin:0;font-size:1.25rem">🎙️ ${escHtml(datos.nombre_original)}</h2>
        <div style="font-size:0.84rem;color:var(--muted);margin-top:2px">
          Duración analizada: <strong>${fmtHMS(datos.duracion)}</strong> · Modelo: <strong>Whisper ${escHtml(datos.modelo)}</strong> ·
          Audio base: <a href="/audios/${encodeURIComponent(datos.nombre_mp3)}" download>${escHtml(datos.nombre_mp3)}</a>
        </div>
      </div>
      <div style="display:flex;align-items:center;gap:8px">
        <label style="font-size:0.83rem;font-weight:700">Re-agrupar voces:</label>
        <select id="recluster-sel-${sid}" style="width:auto;padding:5px 8px;font-size:0.84rem">
          <option value="0">Automático</option>
          <option value="1">1 hablante</option>
          <option value="2">2 hablantes</option>
          <option value="3">3 hablantes</option>
          <option value="4">4 hablantes</option>
          <option value="5">5 hablantes</option>
          <option value="6">6 hablantes</option>
        </select>
        <button type="button" class="btn-outline btn-sm" id="btn-recluster-${sid}">🔄 Recalcular</button>
      </div>
    </div>

    <!-- Barra de Reproducción Fija (Sticky) con Indicadores de Estado -->
    <div class="player-sticky">
      <audio id="audio-${sid}" controls preload="metadata" src="/audios/${encodeURIComponent(datos.nombre_mp3)}"></audio>
      <div class="player-chips">
        <span class="player-chip" id="chip-voz-${sid}">⭐ Voz: Sin asignar</span>
        <span class="player-chip" id="chip-rango-${sid}">📏 Rango: Completo</span>
      </div>
      <div class="player-info" id="pinfo-${sid}">Elegí tu voz con <strong>"⭐ Es mi voz"</strong> o marcá frases de recorte para comenzar.</div>
    </div>

    <!-- Barra de Pestañas (Workflow organizado) -->
    <div class="tabs-nav" id="tabs-nav-${sid}">
      <button type="button" class="tab-btn activo" data-tab="hablantes">
        👥 1. Hablantes & Mi Voz <span class="tab-badge" id="badge-tab-spk-${sid}">${datos.hablantes.length}</span>
      </button>
      <button type="button" class="tab-btn" data-tab="recorte">
        ✂️ 2. Recorte por Frases & Tiempo <span class="tab-badge" id="badge-tab-cut-${sid}">Auto</span>
      </button>
      <button type="button" class="tab-btn" data-tab="transcripcion">
        📝 3. Transcripción <span class="tab-badge" id="badge-tab-trans-${sid}">${datos.segmentos.length}</span>
      </button>
      <button type="button" class="tab-btn" data-tab="exportar">
        💾 4. Exportar Audio & Subtítulos
      </button>
    </div>

    <!-- PESTAÑA 1: HABLANTES Y MI VOZ -->
    <div class="tab-pane activo" data-pane="hablantes">
      <div class="hablantes-intro-banner">
        <div>
          <p>🎯 <strong>Objetivo de clonación:</strong> Hacé clic en <strong>"⭐ Es mi voz"</strong> en tu tarjeta para dejarte a vos como único hablante activo y silenciar automáticamente a los demás.</p>
        </div>
        <div style="display:flex;gap:6px">
          <button type="button" class="btn-outline btn-xs" id="btn-silenciar-todos-${sid}">🔇 Silenciar a todos</button>
          <button type="button" class="btn-outline btn-xs" id="btn-activar-todos-${sid}">🔊 Activar a todos</button>
        </div>
      </div>

      <!-- Memoria de Voces Conocidas -->
      <div class="panel-memoria">
        <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px">
          <div>
            <strong style="font-size:0.92rem;color:#0f172a">🧠 Registro de Voces Conocidas (Memoria Inteligente)</strong>
            <span style="font-size:0.82rem;color:var(--muted);margin-left:6px" id="lbl-voces-reg-cont">Cargando perfiles…</span>
          </div>
          <button type="button" class="btn-outline btn-xs" id="btn-toggle-panel-voces">Ver voces guardadas</button>
        </div>
        <div id="panel-voces-lista" style="display:none;margin-top:10px;padding-top:10px;border-top:1px dashed #cbd5e1">
          <div style="font-size:0.8rem;color:var(--muted);margin-bottom:8px">
            Las voces guardadas se usarán para reconocer y auto-etiquetar a las personas en grabaciones futuras. Podés guardar la voz de cada hablante haciendo clic en <strong>"💾 Guardar voz"</strong> en su tarjeta.
          </div>
          <div id="lista-voces-tags" style="display:flex;flex-wrap:wrap;gap:8px"></div>
        </div>
      </div>

      <!-- Tarjetas de Hablantes -->
      <div class="hablantes-grid" id="hablantes-grid-${sid}"></div>

      <div style="background:#f1f5f9;border-radius:8px;padding:12px;display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:10px">
        <span style="font-size:0.85rem;color:#334155">¿Ya elegiste tu voz? Podés definir marcas de inicio y fin en el recorte o ir directo a exportar.</span>
        <button type="button" class="btn-primary btn-sm btn-nav-goto" data-goto="exportar">👉 Ir al paso 4: Exportar Audio Limpio</button>
      </div>
    </div>

    <!-- PESTAÑA 2: RECORTE POR FRASES Y TIEMPO -->
    <div class="tab-pane" data-pane="recorte">
      <div class="box-recorte">
        <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;margin-bottom:10px">
          <div style="font-weight:700;font-size:0.98rem;color:#1e3a8a">
            🔎 Detección Automática por Frases Habladas (Inicio y Fin)
          </div>
          <div style="font-size:0.8rem;color:#475569">
            El sistema busca las marcas de tiempo exactas donde pronunciás estas frases.
          </div>
        </div>

        <div class="sub-grid-2" style="margin-bottom:12px">
          <div class="campo">
            <label for="f-ini-${sid}">⛳ Cuando digo la frase de inicio:</label>
            <input type="text" id="f-ini-${sid}" value="${escHtml(rd.frase_ini || '')}" placeholder='ej. "bueno empezamos con la clase"'>
            <div id="pill-ini-${sid}"></div>
          </div>
          <div class="campo">
            <label for="f-fin-${sid}">🏁 Cuando finalizo con la frase:</label>
            <input type="text" id="f-fin-${sid}" value="${escHtml(rd.frase_fin || '')}" placeholder='ej. "hasta acá llegamos hoy"'>
            <div id="pill-fin-${sid}"></div>
          </div>
        </div>

        <div style="display:flex;flex-wrap:wrap;gap:8px">
          <button type="button" class="btn-primary btn-sm" id="btn-buscar-frases-${sid}">
            🔍 Detectar marcas de tiempo por frases
          </button>
          <button type="button" class="btn-outline btn-sm" id="btn-limpiar-rango-${sid}">
            ✕ Quitar recorte (usar audio completo)
          </button>
        </div>
      </div>

      <div class="box-recorte" style="background:#ffffff">
        <div style="font-weight:700;font-size:0.95rem;color:#1e293b;margin-bottom:10px">
          ⏱️ Ajuste Manual del Rango Activo y Descarga Rápida
        </div>
        <div style="display:flex;flex-wrap:wrap;align-items:center;gap:12px">
          <div style="display:flex;align-items:center;gap:6px">
            <label style="font-size:0.84rem;font-weight:700">Desde:</label>
            <input type="text" id="t-ini-man-${sid}" placeholder="00:00" style="width:90px;padding:6px 8px;font-family:monospace;font-size:0.88rem">
          </div>
          <div style="display:flex;align-items:center;gap:6px">
            <label style="font-size:0.84rem;font-weight:700">Hasta:</label>
            <input type="text" id="t-fin-man-${sid}" placeholder="${fmtHMS(datos.duracion)}" style="width:90px;padding:6px 8px;font-family:monospace;font-size:0.88rem">
          </div>
          <button type="button" class="btn-outline btn-sm" id="btn-preview-rango-${sid}">▶ Escuchar inicio del recorte</button>
          <button type="button" class="btn-secondary btn-sm" id="btn-export-rango-directo-${sid}">
            ✂️ Descargar este tramo directo (MP3)
          </button>
        </div>
        <p style="font-size:0.8rem;color:var(--muted);margin:8px 0 0">
          💡 Tip: También podés hacer clic en los botones <strong>⛳ Inicio</strong> o <strong>🏁 Fin</strong> en cualquier línea de la pestaña <strong>Transcripción</strong> para fijar los límites.
        </p>
      </div>
    </div>

    <!-- PESTAÑA 3: TRANSCRIPCIÓN INTERACTIVA -->
    <div class="tab-pane" data-pane="transcripcion">
      <div class="trans-toolbar">
        <div style="display:flex;align-items:center;gap:8px">
          <strong style="font-size:0.95rem">Fragmentos:</strong>
          <span class="badge" id="cont-seg-${sid}">${datos.segmentos.length}</span>
        </div>
        <div style="display:flex;gap:8px;flex-wrap:wrap">
          <input type="text" id="buscar-${sid}" placeholder="🔍 Filtrar texto…" style="width:190px;padding:6px 10px;font-size:0.84rem">
          <select id="filtro-spk-${sid}" style="width:165px;padding:6px 10px;font-size:0.84rem">
            <option value="">Todos los hablantes</option>
          </select>
          <button type="button" class="btn-outline btn-sm" id="btn-nuevo-spk-${sid}">➕ Nuevo hablante</button>
        </div>
      </div>
      <div class="segmentos-wrap" id="seg-wrap-${sid}"></div>
    </div>

    <!-- PESTAÑA 4: EXPORTAR AUDIO Y SUBTÍTULOS -->
    <div class="tab-pane" data-pane="exportar">
      <!-- Barra de Presets Rápidos -->
      <div style="margin-bottom:12px;font-size:0.86rem;font-weight:700;color:#334155">
        ⚡ Presets Rápidos de Exportación:
      </div>
      <div class="export-presets-bar">
        <div class="preset-card" id="preset-entreno-${sid}">
          <strong>🎯 Entrenar Voz (XTTS / RVC)</strong>
          <span>Aísla solo tu voz limpia en WAV sin pérdida y descarta frases menores a 1 seg.</span>
        </div>
        <div class="preset-card" id="preset-limpio-${sid}">
          <strong>✂️ Audio Silenciado (MP3)</strong>
          <span>Mantiene el audio completo pero silencia a los demás hablantes.</span>
        </div>
        <div class="preset-card" id="preset-recorte-${sid}">
          <strong>📏 Solo Tramo Recortado</strong>
          <span>Descarga exactamente el lapso entre la marca de inicio y la de fin.</span>
        </div>
      </div>

      <div class="export-grid">
        <!-- Panel de Audio -->
        <div class="export-box">
          <h3>🎯 Exportar Audio con mi Voz Limpia</h3>
          <div class="campo" style="margin-bottom:10px">
            <label for="modo-mute-${sid}">Modo de extracción</label>
            <select id="modo-mute-${sid}">
              <option value="aislar" selected>🎯 Aislar SOLO la voz activa (Ideal entrenamiento de voz: concatena solo tus frases limpias)</option>
              <option value="recortar">✂️ Recortar fragmentos silenciados (acorta el audio eliminando voces ajenas)</option>
              <option value="silenciar">🔇 Silenciar a volumen 0 (conserva la duración original)</option>
              <option value="rango_directo">📏 Solo recortar entre marca de inicio y fin (sin filtrar hablantes)</option>
            </select>
          </div>

          <div class="sub-grid-2" id="opciones-entreno-${sid}">
            <div class="campo">
              <label for="max-min-${sid}">Duración máxima a extraer</label>
              <select id="max-min-${sid}">
                <option value="0" selected>Todo el tiempo disponible en el rango</option>
                <option value="1">Los mejores 1 min de mi voz</option>
                <option value="2">Los mejores 2 min de mi voz</option>
                <option value="3">Los mejores 3 min de mi voz</option>
                <option value="5">Los mejores 5 min de mi voz</option>
                <option value="10">Los mejores 10 min de mi voz</option>
              </select>
            </div>
            <div class="campo">
              <label for="min-seg-${sid}">Ignorar frases muy cortas</label>
              <select id="min-seg-${sid}">
                <option value="0">Conservar todas las frases</option>
                <option value="1.0" selected>Ignorar &lt; 1 seg (muletillas: "sí", "claro")</option>
                <option value="2.0">Ignorar &lt; 2 seg (solo frases completas)</option>
              </select>
            </div>
          </div>

          <div class="sub-grid-2">
            <div class="campo">
              <label for="fmt-audio-${sid}">Formato del archivo de salida</label>
              <select id="fmt-audio-${sid}">
                <option value="mp3" selected>MP3 (liviano, universal)</option>
                <option value="wav">WAV 44.1 kHz sin pérdida (máxima fidelidad para entrenar)</option>
                <option value="zip">📦 Dataset ZIP (clips WAV + metadata.csv para XTTS/RVC)</option>
              </select>
            </div>
            <div class="campo">
              <label for="pausa-seg-${sid}">Pausa limpia entre frases</label>
              <select id="pausa-seg-${sid}">
                <option value="0.15">0.15 seg (continua)</option>
                <option value="0.30" selected>0.30 seg (natural, recomendado)</option>
                <option value="0.60">0.60 seg (pausada)</option>
              </select>
            </div>
          </div>

          <div id="resumen-mute-${sid}" style="font-size:0.86rem;color:#1e40af;background:#eff6ff;padding:10px 12px;border-radius:8px;margin-bottom:12px;border:1px solid #bfdbfe"></div>
          <button type="button" class="btn-primary" id="btn-export-mp3-${sid}" style="width:100%;justify-content:center">
            💾 Generar y Descargar Audio Procesado
          </button>
          <div id="res-mp3-${sid}"></div>
        </div>

        <!-- Panel de Subtítulos y Transcripción -->
        <div class="export-box">
          <h3>📄 Transcripción y Subtítulos</h3>
          <p style="font-size:0.84rem;color:var(--muted);margin-top:0">Exportá el texto de la grabación respetando el rango y los hablantes activos.</p>
          <label class="check-inline" style="margin-bottom:8px">
            <input type="checkbox" id="inc-sil-${sid}">
            <span>Incluir fragmentos silenciados en el archivo</span>
          </label>
          <label class="check-inline" style="margin-bottom:16px">
            <input type="checkbox" id="inc-nom-${sid}" checked>
            <span>Incluir nombre del hablante en cada línea</span>
          </label>
          <div style="display:flex;flex-direction:column;gap:8px">
            <button type="button" class="btn-outline btn-sm" data-fmt="srt" style="justify-content:center">
              🎬 Subtítulos SRT (Para YouTube, Premiere o VLC)
            </button>
            <button type="button" class="btn-outline btn-sm" data-fmt="txt_ts" style="justify-content:center">
              ⏱️ TXT con marcas de tiempo [MM:SS]
            </button>
            <button type="button" class="btn-outline btn-sm" data-fmt="txt" style="justify-content:center">
              📝 TXT de texto corrido
            </button>
          </div>
        </div>
      </div>
    </div>
  `;
  zonaEstudios.prepend(card);

  // Manejo de cambio de pestañas (Tabs)
  const navTabs = card.querySelector('#tabs-nav-' + sid);
  function cambiarTab(tabKey) {
    navTabs.querySelectorAll('.tab-btn').forEach(b => b.classList.toggle('activo', b.dataset.tab === tabKey));
    card.querySelectorAll('.tab-pane').forEach(p => p.classList.toggle('activo', p.dataset.pane === tabKey));
  }
  navTabs.querySelectorAll('.tab-btn').forEach(b => {
    b.onclick = () => cambiarTab(b.dataset.tab);
  });
  card.querySelectorAll('.btn-nav-goto').forEach(b => {
    b.onclick = () => cambiarTab(b.dataset.goto);
  });

  // Presets rápidos de exportación
  const pEntreno = document.getElementById('preset-entreno-' + sid);
  if (pEntreno) {
    pEntreno.onclick = () => {
      document.getElementById('modo-mute-' + sid).value = 'aislar';
      document.getElementById('fmt-audio-' + sid).value = 'wav';
      document.getElementById('min-seg-' + sid).value = '1.0';
      document.getElementById('pausa-seg-' + sid).value = '0.30';
      actualizarResumenMute(sid);
    };
  }
  const pLimpio = document.getElementById('preset-limpio-' + sid);
  if (pLimpio) {
    pLimpio.onclick = () => {
      document.getElementById('modo-mute-' + sid).value = 'recortar';
      document.getElementById('fmt-audio-' + sid).value = 'mp3';
      actualizarResumenMute(sid);
    };
  }
  const pRecorte = document.getElementById('preset-recorte-' + sid);
  if (pRecorte) {
    pRecorte.onclick = () => {
      document.getElementById('modo-mute-' + sid).value = 'rango_directo';
      document.getElementById('fmt-audio-' + sid).value = 'mp3';
      actualizarResumenMute(sid);
    };
  }

  const audioEl = document.getElementById('audio-' + sid);
  const pInfo = document.getElementById('pinfo-' + sid);

  audioEl.ontimeupdate = () => {
    const st = sesiones[sid];
    if (st.pararEn !== null && audioEl.currentTime >= st.pararEn) {
      audioEl.pause();
      st.pararEn = null;
      pInfo.textContent = 'Muestra finalizada.';
    }
  };

  // Eventos de búsqueda de frases de inicio y fin
  document.getElementById('btn-buscar-frases-' + sid).onclick = () => detectarMarcasPorFrase(sid);
  document.getElementById('f-ini-' + sid).onkeydown = e => { if (e.key === 'Enter') detectarMarcasPorFrase(sid); };
  document.getElementById('f-fin-' + sid).onkeydown = e => { if (e.key === 'Enter') detectarMarcasPorFrase(sid); };

  document.getElementById('btn-limpiar-rango-' + sid).onclick = () => {
    const st = sesiones[sid];
    st.rangoInicio = null;
    st.rangoFin = null;
    st.matchIni = null;
    st.matchFin = null;
    document.getElementById('f-ini-' + sid).value = '';
    document.getElementById('f-fin-' + sid).value = '';
    sincronizarInputsRango(sid);
    renderSegmentos(sid);
    actualizarResumenMute(sid);
  };

  document.getElementById('t-ini-man-' + sid).onchange = e => {
    sesiones[sid].rangoInicio = parseTiempoCliente(e.target.value);
    sincronizarInputsRango(sid);
    renderSegmentos(sid);
    actualizarResumenMute(sid);
  };
  document.getElementById('t-fin-man-' + sid).onchange = e => {
    sesiones[sid].rangoFin = parseTiempoCliente(e.target.value);
    sincronizarInputsRango(sid);
    renderSegmentos(sid);
    actualizarResumenMute(sid);
  };

  document.getElementById('btn-preview-rango-' + sid).onclick = () => {
    const st = sesiones[sid];
    const ini = st.rangoInicio ?? 0;
    const fin = Math.min(st.duracion, ini + 8);
    reproducirRango(sid, ini, fin, 'Inicio del recorte');
  };

  document.getElementById('btn-export-rango-directo-' + sid).onclick = () => {
    cambiarTab('exportar');
    document.getElementById('modo-mute-' + sid).value = 'rango_directo';
    actualizarResumenMute(sid);
    exportarAudioFiltrado(sid);
  };

  document.getElementById('modo-mute-' + sid).onchange = () => actualizarResumenMute(sid);
  document.getElementById('min-seg-' + sid).onchange = () => actualizarResumenMute(sid);
  document.getElementById('max-min-' + sid).onchange = () => actualizarResumenMute(sid);
  document.getElementById('fmt-audio-' + sid).onchange = () => actualizarResumenMute(sid);

  document.getElementById('buscar-' + sid).oninput = e => {
    sesiones[sid].filtroTexto = e.target.value.toLowerCase();
    renderSegmentos(sid);
  };
  document.getElementById('filtro-spk-' + sid).onchange = e => {
    sesiones[sid].filtroSpk = e.target.value;
    renderSegmentos(sid);
  };
  document.getElementById('btn-nuevo-spk-' + sid).onclick = () => {
    const st = sesiones[sid];
    const num = st.hablantes.length + 1;
    const nuevoId = 'SPK_' + Date.now().toString().slice(-4);
    const colores = ["#2563eb", "#059669", "#d97706", "#7c3aed", "#db2777", "#0891b2", "#dc2626"];
    st.hablantes.push({
      id: nuevoId,
      nombre: `Hablante ${num}`,
      color: colores[(num - 1) % colores.length],
      silenciado: false,
      cantidad: 0,
      duracion: 0,
      muestras: []
    });
    renderHablantes(sid);
    renderSegmentos(sid);
  };

  document.getElementById('btn-recluster-' + sid).onclick = async () => {
    const k = document.getElementById('recluster-sel-' + sid).value;
    const btn = document.getElementById('btn-recluster-' + sid);
    btn.disabled = true;
    try {
      const r = await fetch('/api/reagrupar', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ sesion_id: sid, num_hablantes: Number(k) })
      });
      if (!r.ok) throw new Error(await r.text());
      const d = await r.json();
      sesiones[sid].hablantes = d.hablantes;
      sesiones[sid].segmentos = d.segmentos;
      sesiones[sid].miVozId = null;
      renderHablantes(sid);
      renderSegmentos(sid);
    } catch (e) {
      alert('Error al reagrupar hablantes: ' + e.message);
    } finally {
      btn.disabled = false;
    }
  };

  document.getElementById('btn-export-mp3-' + sid).onclick = () => exportarAudioFiltrado(sid);

  document.getElementById('btn-silenciar-todos-' + sid).onclick = () => {
    const st = sesiones[sid];
    for (const h of st.hablantes) h.silenciado = true;
    for (const s of st.segmentos) delete s.override_mute;
    renderHablantes(sid);
    renderSegmentos(sid);
  };

  document.getElementById('btn-activar-todos-' + sid).onclick = () => {
    const st = sesiones[sid];
    for (const h of st.hablantes) h.silenciado = false;
    for (const s of st.segmentos) delete s.override_mute;
    renderHablantes(sid);
    renderSegmentos(sid);
  };

  for (const b of card.querySelectorAll('[data-fmt]')) {
    b.onclick = () => descargarTranscripcion(sid, b.dataset.fmt);
  }

  // Manejo del panel desplegable de memoria de voces
  const btnToggleVoces = document.getElementById('btn-toggle-panel-voces');
  if (btnToggleVoces) {
    btnToggleVoces.onclick = () => {
      const p = document.getElementById('panel-voces-lista');
      if (p.style.display === 'none') {
        p.style.display = 'block';
        btnToggleVoces.textContent = 'Ocultar voces guardadas';
      } else {
        p.style.display = 'none';
        btnToggleVoces.textContent = 'Ver voces guardadas';
      }
    };
  }

  actualizarListaVocesRegistradasGlobal();
  sincronizarInputsRango(sid);
  renderHablantes(sid);
  renderSegmentos(sid);
}

async function actualizarListaVocesRegistradasGlobal() {
  try {
    const r = await fetch('/api/voces_registradas');
    if (!r.ok) return;
    const perfiles = await r.json();
    const lbl = document.getElementById('lbl-voces-reg-cont');
    const tags = document.getElementById('lista-voces-tags');
    if (lbl) {
      lbl.textContent = perfiles.length === 1 ? '1 persona registrada' : `${perfiles.length} personas registradas`;
    }
    if (tags) {
      if (!perfiles.length) {
        tags.innerHTML = '<span style="font-size:0.82rem;color:var(--muted)">Todavía no registraste ninguna voz. Hacé clic en "💾 Guardar voz" en la tarjeta de cualquier persona para recordarla automáticamente en el futuro.</span>';
        return;
      }
      tags.innerHTML = perfiles.map(p => `
        <span class="badge" style="background:#e0f2fe;color:#0369a1;display:inline-flex;align-items:center;gap:6px;padding:5px 9px;font-size:0.83rem">
          <span style="display:inline-block;width:9px;height:9px;border-radius:50%;background:${escHtml(p.color || '#0284c7')}"></span>
          <strong>${escHtml(p.nombre)}</strong> (${p.n_muestras || 1})
          <button type="button" class="btn-xs btn-del-voz" data-nombre="${escHtml(p.nombre)}" style="background:transparent;border:none;cursor:pointer;color:#dc2626;font-size:0.88rem;padding:0 3px" title="Eliminar de la memoria">✕</button>
        </span>
      `).join('');

      for (const btn of tags.querySelectorAll('.btn-del-voz')) {
        btn.onclick = async () => {
          const nom = btn.dataset.nombre;
          if (!confirm(`¿Eliminar la voz de "${nom}" de la memoria?`)) return;
          await fetch('/api/eliminar_voz_registro', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ nombre: nom })
          });
          actualizarListaVocesRegistradasGlobal();
        };
      }
    }
  } catch (e) {
    console.error('Error cargando voces registradas:', e);
  }
}

function sincronizarInputsRango(sid) {
  const st = sesiones[sid];
  const inIni = document.getElementById('t-ini-man-' + sid);
  const inFin = document.getElementById('t-fin-man-' + sid);
  const pillIni = document.getElementById('pill-ini-' + sid);
  const pillFin = document.getElementById('pill-fin-' + sid);
  const chipRango = document.getElementById('chip-rango-' + sid);
  const badgeCut = document.getElementById('badge-tab-cut-' + sid);

  inIni.value = st.rangoInicio !== null ? fmtHMS(st.rangoInicio) : '';
  inFin.value = st.rangoFin !== null ? fmtHMS(st.rangoFin) : '';

  if (chipRango) {
    if (st.rangoInicio !== null || st.rangoFin !== null) {
      const r0 = st.rangoInicio ?? 0;
      const r1 = st.rangoFin ?? st.duracion;
      chipRango.className = 'player-chip destacado';
      chipRango.innerHTML = `✂️ Recorte: [${fmtHMS(r0)} → ${fmtHMS(r1)}]`;
      if (badgeCut) badgeCut.textContent = `${fmtHMS(r0)} → ${fmtHMS(r1)}`;
    } else {
      chipRango.className = 'player-chip';
      chipRango.innerHTML = `📏 Audio completo (${fmtHMS(st.duracion)})`;
      if (badgeCut) badgeCut.textContent = 'Completo';
    }
  }

  if (st.matchIni) {
    pillIni.innerHTML = `
      <div class="match-pill">
        <span>✔ Detectado en <strong>${fmtHMS(st.matchIni.frase_inicio)}</strong>: <em>"${escHtml(st.matchIni.texto.slice(0, 65))}"</em></span>
        <button type="button" class="btn-outline btn-xs" id="play-m-ini-${sid}">▶</button>
      </div>
    `;
    document.getElementById('play-m-ini-' + sid).onclick = () =>
      reproducirRango(sid, st.matchIni.seg_inicio, st.matchIni.seg_fin, 'Frase de inicio');
  } else {
    pillIni.innerHTML = '';
  }

  if (st.matchFin) {
    pillFin.innerHTML = `
      <div class="match-pill">
        <span>✔ Detectado en <strong>${fmtHMS(st.matchFin.frase_fin)}</strong>: <em>"${escHtml(st.matchFin.texto.slice(0, 65))}"</em></span>
        <button type="button" class="btn-outline btn-xs" id="play-m-fin-${sid}">▶</button>
      </div>
    `;
    document.getElementById('play-m-fin-' + sid).onclick = () =>
      reproducirRango(sid, st.matchFin.seg_inicio, st.matchFin.seg_fin, 'Frase de cierre');
  } else {
    pillFin.innerHTML = '';
  }
}

async function detectarMarcasPorFrase(sid) {
  const st = sesiones[sid];
  const fIni = document.getElementById('f-ini-' + sid).value.trim();
  const fFin = document.getElementById('f-fin-' + sid).value.trim();
  if (!fIni && !fFin) {
    alert('Escribí al menos una frase de inicio o una frase de cierre para buscar su marca de tiempo.');
    return;
  }

  try {
    const r = await fetch('/api/buscar_frases', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        sesion_id: sid,
        frase_ini: fIni,
        frase_fin: fFin,
        segmentos: st.segmentos
      })
    });
    if (!r.ok) throw new Error(await r.text());
    const d = await r.json();

    if (fIni) {
      if (d.match_ini) {
        st.rangoInicio = d.match_ini.frase_inicio;
        st.matchIni = d.match_ini;
      } else {
        st.matchIni = null;
        alert(`No encontré una frase parecida a "${fIni}" en la transcripción.`);
      }
    }
    if (fFin) {
      if (d.match_fin) {
        st.rangoFin = d.match_fin.frase_fin;
        st.matchFin = d.match_fin;
      } else {
        st.matchFin = null;
        alert(`No encontré una frase parecida a "${fFin}" después del inicio.`);
      }
    }
    sincronizarInputsRango(sid);
    renderSegmentos(sid);
    actualizarResumenMute(sid);
  } catch (e) {
    alert('Error buscando frases: ' + e.message);
  }
}

function reproducirRango(sid, inicio, fin, etiqueta) {
  const st = sesiones[sid];
  const audioEl = document.getElementById('audio-' + sid);
  const pInfo = document.getElementById('pinfo-' + sid);
  st.pararEn = fin;
  audioEl.currentTime = Math.max(0, inicio);
  audioEl.play();
  pInfo.textContent = `▶ Escuchando: ${etiqueta} (${fmtHMS(inicio)} → ${fmtHMS(fin)})`;
}

function recalcularEstadisticasHablantes(sid) {
  const st = sesiones[sid];
  for (const h of st.hablantes) {
    const segs = st.segmentos.filter(s => s.hablante === h.id);
    h.cantidad = segs.length;
    h.duracion = Math.round(segs.reduce((acc, s) => acc + Math.max(0, s.fin - s.inicio), 0) * 10) / 10;
    const ordenados = [...segs].sort((a, b) => (b.fin - b.inicio) - (a.fin - a.inicio));
    h.muestras = ordenados.slice(0, 5).map(m => ({ inicio: m.inicio, fin: m.fin, texto: m.texto.slice(0, 90) }));
  }
}

function esSegmentoSilenciado(st, seg) {
  if (seg.override_mute !== undefined && seg.override_mute !== null) {
    return Boolean(seg.override_mute);
  }
  const h = st.hablantes.find(x => x.id === seg.hablante);
  return Boolean(h && h.silenciado);
}

function estaDentroDelRango(st, seg) {
  const r0 = st.rangoInicio ?? 0;
  const r1 = st.rangoFin ?? st.duracion;
  return seg.fin > r0 && seg.inicio < r1;
}

function actualizarResumenMute(sid) {
  const st = sesiones[sid];
  const modo = document.getElementById('modo-mute-' + sid)?.value || 'aislar';
  const minSeg = Number(document.getElementById('min-seg-' + sid)?.value || 0);
  const maxMin = Number(document.getElementById('max-min-' + sid)?.value || 0);
  const boxEntreno = document.getElementById('opciones-entreno-' + sid);
  if (boxEntreno) {
    boxEntreno.style.display = modo === 'aislar' ? 'grid' : 'none';
  }

  const r0 = st.rangoInicio ?? 0;
  const r1 = st.rangoFin ?? st.duracion;
  const durRango = Math.max(0, r1 - r0);
  const txtRango = (st.rangoInicio !== null || st.rangoFin !== null)
    ? ` dentro del recorte <strong>[${fmtHMS(r0)} → ${fmtHMS(r1)}]</strong>`
    : '';

  const segsEnRango = st.segmentos.filter(s => estaDentroDelRango(st, s));
  let activos = segsEnRango.filter(s => !esSegmentoSilenciado(st, s));
  if (modo === 'aislar' && minSeg > 0) {
    activos = activos.filter(s => (Math.min(s.fin, r1) - Math.max(s.inicio, r0)) >= minSeg);
  }
  let durActiva = activos.reduce((acc, s) => acc + Math.max(0, Math.min(s.fin, r1) - Math.max(s.inicio, r0)), 0);
  if (modo === 'aislar' && maxMin > 0 && durActiva > maxMin * 60) {
    durActiva = maxMin * 60;
  }

  const silenciados = segsEnRango.filter(s => esSegmentoSilenciado(st, s));
  const durSil = silenciados.reduce((acc, s) => acc + Math.max(0, Math.min(s.fin, r1) - Math.max(s.inicio, r0)), 0);

  const el = document.getElementById('resumen-mute-' + sid);
  if (modo === 'rango_directo') {
    el.innerHTML = `📏 <strong>Recorte directo:</strong> desde <strong>${fmtHMS(r0)}</strong> hasta <strong>${fmtHMS(r1)}</strong> (Duración estimada: <strong>${fmtHMS(durRango)}</strong>).`;
  } else if (modo === 'aislar') {
    el.innerHTML = `✨ <strong>Voz activa lista para extraer${txtRango}:</strong> ${activos.length} frases limpias · Duración estimada: <strong>${fmtHMS(durActiva)}</strong>.`;
  } else {
    el.innerHTML = `🔇 Se silenciarán <strong>${silenciados.length} fragmentos</strong> (<strong>${fmtHMS(durSil)}</strong>)${txtRango} y quedarán <strong>${fmtHMS(Math.max(0, durRango - durSil))}</strong>.`;
  }
}

function renderHablantes(sid) {
  const st = sesiones[sid];
  recalcularEstadisticasHablantes(sid);
  const grid = document.getElementById('hablantes-grid-' + sid);
  const selFiltro = document.getElementById('filtro-spk-' + sid);
  const chipVoz = document.getElementById('chip-voz-' + sid);
  const badgeSpk = document.getElementById('badge-tab-spk-' + sid);
  const filtroActual = selFiltro.value;

  selFiltro.innerHTML = '<option value="">Todos los hablantes</option>';
  grid.innerHTML = '';

  const miHablante = st.hablantes.find(x => x.id === st.miVozId && !x.silenciado);
  if (chipVoz) {
    if (miHablante) {
      chipVoz.className = 'player-chip destacado';
      chipVoz.innerHTML = `⭐ Mi Voz: <strong>${escHtml(miHablante.nombre)}</strong>`;
      if (badgeSpk) badgeSpk.textContent = `⭐ ${miHablante.nombre}`;
    } else {
      chipVoz.className = 'player-chip';
      chipVoz.innerHTML = '⚪ Voz: Sin asignar';
      if (badgeSpk) badgeSpk.textContent = `${st.hablantes.length}`;
    }
  }

  for (const h of st.hablantes) {
    const opt = document.createElement('option');
    opt.value = h.id;
    opt.textContent = `${h.nombre} (${h.cantidad})`;
    selFiltro.appendChild(opt);

    const esMiVoz = st.miVozId === h.id && !h.silenciado;
    const esAuto = Boolean(h.auto_etiquetado);
    const card = document.createElement('div');
    card.className = 'spk-card' + (h.silenciado ? ' silenciado' : '') + (esMiVoz ? ' es-mi-voz' : '');
    card.innerHTML = `
      <div class="spk-top">
        <span class="spk-dot" style="background:${h.color}"></span>
        <input type="text" class="spk-name-input" value="${escHtml(h.nombre)}" title="Hacé clic para renombrar a este hablante">
        ${esAuto ? '<span class="badge badge-azul" title="Reconocido automáticamente desde el registro de voces">🧠 Reconocido</span>' : ''}
        ${esMiVoz ? '<span class="badge badge-verde">⭐ Tu Voz</span>' : ''}
      </div>
      <div class="spk-meta">
        ${h.cantidad} intervenciones · <strong>${fmtHMS(h.duracion)}</strong> hablados
      </div>
      <div class="spk-actions">
        <button type="button" class="btn-outline btn-sm btn-muestra" ${h.muestras.length ? '' : 'disabled'}>
          ▶ Ejemplo
        </button>
        <button type="button" class="btn-gold btn-aislar ${esMiVoz ? 'activo' : ''}" title="Dejar activa SOLO esta voz y silenciar a todos los demás">
          ${esMiVoz ? '✔ Es mi voz' : '⭐ Es mi voz'}
        </button>
        <button type="button" class="btn-outline btn-sm btn-guardar-voz" title="Guardar la huella acústica de esta persona para auto-etiquetarla en grabaciones futuras">
          💾 Guardar
        </button>
        <label class="mute-toggle">
          <input type="checkbox" class="chk-mute" ${h.silenciado ? 'checked' : ''} hidden>
          <span>${h.silenciado ? '🔇 Silenciado' : '🔊 Activo'}</span>
        </label>
      </div>
    `;

    const inputNombre = card.querySelector('.spk-name-input');
    inputNombre.onchange = () => {
      h.nombre = inputNombre.value.trim() || h.id;
      renderHablantes(sid);
      renderSegmentos(sid);
    };

    const btnGuardarVoz = card.querySelector('.btn-guardar-voz');
    btnGuardarVoz.onclick = async () => {
      const nom = h.nombre.trim();
      if (!nom || nom.startsWith('Hablante ')) {
        const nuevoNom = prompt('Ingresá el nombre real de esta persona para guardarla en el registro:', nom.startsWith('Hablante ') ? '' : nom);
        if (!nuevoNom || !nuevoNom.trim()) return;
        h.nombre = nuevoNom.trim();
      }
      btnGuardarVoz.disabled = true;
      btnGuardarVoz.textContent = 'Guardando…';
      try {
        const r = await fetch('/api/guardar_voz_registro', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({
            sesion_id: sid,
            spk_id: h.id,
            nombre: h.nombre,
            color: h.color
          })
        });
        if (!r.ok) throw new Error(await r.text());
        alert(`¡Listo! La voz de "${h.nombre}" quedó registrada. Cuando transcribas nuevas grabaciones donde hable, se etiquetará automáticamente.`);
        actualizarListaVocesRegistradasGlobal();
        renderHablantes(sid);
      } catch (e) {
        alert('Error al registrar voz: ' + e.message);
      } finally {
        btnGuardarVoz.disabled = false;
        btnGuardarVoz.textContent = '💾 Guardar';
      }
    };

    const btnMuestra = card.querySelector('.btn-muestra');
    btnMuestra.onclick = () => {
      if (!h.muestras.length) return;
      const idx = (st.muestraIdx[h.id] || 0) % h.muestras.length;
      st.muestraIdx[h.id] = idx + 1;
      const m = h.muestras[idx];
      btnMuestra.textContent = h.muestras.length > 1 ? `⏭ Ejemplo (${idx + 1}/${h.muestras.length})` : '▶ Ejemplo';
      reproducirRango(sid, m.inicio, m.fin, `${h.nombre}: "${m.texto}"`);
    };

    card.querySelector('.btn-aislar').onclick = () => {
      st.miVozId = h.id;
      if (h.nombre.startsWith('Hablante ')) {
        h.nombre = 'Mi voz (' + h.nombre + ')';
      }
      for (const otro of st.hablantes) {
        otro.silenciado = (otro.id !== h.id);
      }
      for (const s of st.segmentos) {
        delete s.override_mute;
      }
      document.getElementById('modo-mute-' + sid).value = 'aislar';
      renderHablantes(sid);
      renderSegmentos(sid);
    };

    const chkMute = card.querySelector('.chk-mute');
    chkMute.onchange = () => {
      h.silenciado = chkMute.checked;
      if (h.silenciado && st.miVozId === h.id) st.miVozId = null;
      for (const s of st.segmentos) {
        if (s.hablante === h.id) delete s.override_mute;
      }
      renderHablantes(sid);
      renderSegmentos(sid);
    };

    grid.appendChild(card);
  }

  selFiltro.value = filtroActual;
  actualizarResumenMute(sid);
}

function renderSegmentos(sid) {
  const st = sesiones[sid];
  const wrap = document.getElementById('seg-wrap-' + sid);
  wrap.innerHTML = '';

  const mapaHablantes = {};
  for (const h of st.hablantes) mapaHablantes[h.id] = h;

  const filtrados = st.segmentos.filter(s => {
    if (st.filtroSpk && s.hablante !== st.filtroSpk) return false;
    if (st.filtroTexto && !s.texto.toLowerCase().includes(st.filtroTexto)) return false;
    return true;
  });

  document.getElementById('cont-seg-' + sid).textContent = filtrados.length;
  const badgeTrans = document.getElementById('badge-tab-trans-' + sid);
  if (badgeTrans) badgeTrans.textContent = `${filtrados.length}`;

  for (const seg of filtrados) {
    const sil = esSegmentoSilenciado(st, seg);
    const enRango = estaDentroDelRango(st, seg);
    const spk = mapaHablantes[seg.hablante] || { nombre: seg.hablante, color: '#6b7280' };

    const row = document.createElement('div');
    let cls = 'seg-row';
    if (sil) cls += ' seg-silenciado';
    if (!enRango) cls += ' fuera-de-rango';
    if (st.rangoInicio !== null && Math.abs(seg.inicio - st.rangoInicio) < 1.5) cls += ' marca-inicio';
    if (st.rangoFin !== null && Math.abs(seg.fin - st.rangoFin) < 1.5) cls += ' marca-fin';
    row.className = cls;

    const opcionesSpk = st.hablantes.map(h =>
      `<option value="${h.id}" ${h.id === seg.hablante ? 'selected' : ''}>${escHtml(h.nombre)}</option>`
    ).join('');

    row.innerHTML = `
      <button type="button" class="seg-time-btn" title="Escuchar este fragmento">
        ▶ ${fmtHMS(seg.inicio)} - ${fmtHMS(seg.fin)}
      </button>
      <select class="seg-spk-select" style="border-left: 4px solid ${spk.color}">
        ${opcionesSpk}
      </select>
      <div class="seg-texto" contenteditable="true" spellcheck="false">${escHtml(seg.texto)}</div>
      <div style="display:flex;gap:4px;align-items:center">
        <button type="button" class="btn-outline btn-xs btn-set-ini" title="Fijar inicio del recorte acá (${fmtHMS(seg.inicio)})">⛳ Inicio</button>
        <button type="button" class="btn-outline btn-xs btn-set-fin" title="Fijar fin del recorte acá (${fmtHMS(seg.fin)})">🏁 Fin</button>
        <button type="button" class="btn-outline btn-xs btn-seg-mute" title="Silenciar o activar este fragmento puntual">
          ${sil ? '🔇' : '🔊'}
        </button>
      </div>
    `;

    row.querySelector('.seg-time-btn').onclick = () => {
      reproducirRango(sid, seg.inicio, seg.fin, `${spk.nombre}`);
    };

    const selSpk = row.querySelector('.seg-spk-select');
    selSpk.onchange = () => {
      seg.hablante = selSpk.value;
      delete seg.override_mute;
      renderHablantes(sid);
      renderSegmentos(sid);
    };

    const divTxt = row.querySelector('.seg-texto');
    divTxt.onblur = () => {
      seg.texto = divTxt.textContent.trim();
    };

    row.querySelector('.btn-set-ini').onclick = () => {
      st.rangoInicio = seg.inicio;
      st.matchIni = { seg_inicio: seg.inicio, seg_fin: seg.fin, frase_inicio: seg.inicio, frase_fin: seg.fin, texto: seg.texto };
      sincronizarInputsRango(sid);
      renderSegmentos(sid);
      actualizarResumenMute(sid);
    };

    row.querySelector('.btn-set-fin').onclick = () => {
      st.rangoFin = seg.fin;
      st.matchFin = { seg_inicio: seg.inicio, seg_fin: seg.fin, frase_inicio: seg.inicio, frase_fin: seg.fin, texto: seg.texto };
      sincronizarInputsRango(sid);
      renderSegmentos(sid);
      actualizarResumenMute(sid);
    };

    row.querySelector('.btn-seg-mute').onclick = () => {
      seg.override_mute = !sil;
      renderSegmentos(sid);
      actualizarResumenMute(sid);
    };

    wrap.appendChild(row);
  }
}

async function exportarAudioFiltrado(sid) {
  const st = sesiones[sid];
  const btn = document.getElementById('btn-export-mp3-' + sid);
  const resBox = document.getElementById('res-mp3-' + sid);
  const modo = document.getElementById('modo-mute-' + sid).value;
  const formato = document.getElementById('fmt-audio-' + sid).value;
  const maxMin = Number(document.getElementById('max-min-' + sid).value || 0);
  const minSeg = Number(document.getElementById('min-seg-' + sid).value || 0);
  const pausaSeg = Number(document.getElementById('pausa-seg-' + sid).value || 0.3);

  const intervalosSilenciados = st.segmentos
    .filter(s => esSegmentoSilenciado(st, s))
    .map(s => [s.inicio, s.fin]);

  const intervalosActivos = st.segmentos
    .filter(s => !esSegmentoSilenciado(st, s))
    .map(s => ({
      inicio: s.inicio,
      fin: s.fin,
      texto: s.texto,
      confianza: s.confianza || 0.8
    }));

  if (modo === 'aislar' || formato === 'zip') {
    if (!intervalosActivos.length) {
      alert('No hay ningún hablante o frase activa para extraer. Desmarcá al menos tu voz.');
      return;
    }
  } else if (modo !== 'rango_directo' && !intervalosSilenciados.length && st.rangoInicio === null && st.rangoFin === null) {
    alert('Marcá al menos un hablante con 🔇, o definí una marca de inicio/fin, o usá el modo "Aislar SOLO la voz activa".');
    return;
  }

  btn.disabled = true;
  resBox.innerHTML = '<div style="margin-top:10px;font-size:0.88rem">⏳ Generando archivo de audio…</div>';

  try {
    const r = await fetch('/api/exportar_mp3', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        sesion_id: sid,
        modo,
        formato,
        max_minutos: maxMin,
        min_dur_seg: minSeg,
        pausa_seg: pausaSeg,
        rango_inicio: st.rangoInicio,
        rango_fin: st.rangoFin,
        intervalos: intervalosSilenciados,
        activos: intervalosActivos
      })
    });
    if (!r.ok) throw new Error(await r.text());
    const d = await r.json();
    const esAudioReproducible = d.nombre_archivo.endsWith('.mp3') || d.nombre_archivo.endsWith('.wav');
    resBox.innerHTML = `
      <div class="resultado-mp3-box">
        <div style="font-weight:700;color:var(--ok);margin-bottom:6px">
          ✔ Listo: ${escHtml(d.nombre_archivo)} · Duración resultante: ${fmtHMS(d.duracion_final)}
        </div>
        ${esAudioReproducible ? `<audio controls src="/audios/${encodeURIComponent(d.nombre_archivo)}" style="width:100%;height:34px;margin-bottom:8px"></audio>` : ''}
        <a class="btn-primary btn-sm" style="text-decoration:none" href="/audios/${encodeURIComponent(d.nombre_archivo)}" download>
          ⬇ Descargar ${escHtml(d.nombre_archivo)}
        </a>
      </div>
    `;
    cargarListaAudiosExistentes();
  } catch (e) {
    resBox.innerHTML = `<div style="margin-top:10px;color:var(--danger);font-weight:600">✘ ${escHtml(e.message)}</div>`;
  } finally {
    btn.disabled = false;
  }
}

function fmtSrtTs(seg) {
  const msTotal = Math.max(0, Math.round(seg * 1000));
  const h = Math.floor(msTotal / 3600000);
  const m = Math.floor((msTotal % 3600000) / 60000);
  const s = Math.floor((msTotal % 60000) / 1000);
  const ms = msTotal % 1000;
  return `${String(h).padStart(2,'0')}:${String(m).padStart(2,'0')}:${String(s).padStart(2,'0')},${String(ms).padStart(3,'0')}`;
}

function descargarTranscripcion(sid, formato) {
  const st = sesiones[sid];
  const incSil = document.getElementById('inc-sil-' + sid).checked;
  const incNom = document.getElementById('inc-nom-' + sid).checked;
  const mapaNom = {};
  for (const h of st.hablantes) mapaNom[h.id] = h.nombre;

  const segs = st.segmentos.filter(s => estaDentroDelRango(st, s) && (incSil || !esSegmentoSilenciado(st, s)));
  let contenido = '';
  let ext = '.txt';

  if (formato === 'txt_ts') {
    contenido = segs.map(s => {
      const pref = incNom ? `${mapaNom[s.hablante] || s.hablante}: ` : '';
      return `[${fmtHMS(s.inicio)} - ${fmtHMS(s.fin)}] ${pref}${s.texto}`;
    }).join('\n');
    ext = '_marcas.txt';
  } else if (formato === 'txt') {
    contenido = segs.map(s => {
      const pref = incNom ? `${mapaNom[s.hablante] || s.hablante}: ` : '';
      return `${pref}${s.texto}`;
    }).join('\n\n');
    ext = '.txt';
  } else if (formato === 'srt') {
    contenido = segs.map((s, i) => {
      const pref = incNom ? `[${mapaNom[s.hablante] || s.hablante}] ` : '';
      return `${i + 1}\n${fmtSrtTs(s.inicio)} --> ${fmtSrtTs(s.fin)}\n${pref}${s.texto}\n`;
    }).join('\n');
    ext = '.srt';
  }

  const blob = new Blob([contenido], { type: 'text/plain;charset=utf-8' });
  const a = document.createElement('a');
  const base = st.nombre_original.replace(/\.[^.]+$/, '');
  a.href = URL.createObjectURL(blob);
  a.download = `${base}${ext}`;
  a.click();
}
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
        ctype = "audio/wav" if ext == ".wav" else ("application/zip" if ext == ".zip" else "audio/mpeg")
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
            # Si se pasa ?verificar=1 fuerza fetch y check de estado
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
                    if f.lower().endswith((".mp3", ".wav")):
                        rp = os.path.join(SALIDA, f)
                        if os.path.isfile(rp):
                            archivos.append({
                                "nombre": f,
                                "bytes": os.path.getsize(rp),
                                "mtime": os.path.getmtime(rp)
                            })
            archivos.sort(key=lambda x: x["mtime"], reverse=True)
            return self._send_json(200, archivos)

        if p.path == "/api/voces_registradas":
            reg = cargar_registro_voces()
            # Devolver resumen de perfiles sin el vector gigante
            resumen = []
            for nom, info in reg.get("perfiles", {}).items():
                resumen.append({
                    "nombre": nom,
                    "color": info.get("color", "#2563eb"),
                    "n_muestras": info.get("n_muestras", 1),
                    "actualizado_en": info.get("actualizado_en") or info.get("creado_en", "")
                })
            return self._send_json(200, resumen)

        if p.path == "/api/estado":
            q = urllib.parse.parse_qs(p.query)
            jid = q.get("id", [""])[0]
            with JOBS_LOCK:
                job = JOBS.get(jid)
                if not job:
                    return self._send_text(404, "Trabajo no encontrado")
                copia = dict(job)
            return self._send_json(200, copia)

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
            accion = body.get("accion", "auto")  # "auto", "pull", "push", "configurar_remote"
            url_remote = (body.get("remote_url") or "").strip()

            if accion == "configurar_remote" and url_remote:
                code_r, _, _ = ejecutar_git("remote", "remove", "origin")
                code_a, _, err_a = ejecutar_git("remote", "add", "origin", url_remote)
                if code_a != 0:
                    return self._send_text(400, f"Error al configurar remote: {err_a}")
                # Hacer commit de archivos si hay cambios pendientes antes de sincronizar
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

            # Accion auto
            res = verificar_y_sincronizar_git(accion_automatica=True)
            return self._send_json(200, res)

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
                sesion["segmentos"], sesion["X"], num_hablantes=num_hablantes
            )
            sesion["segmentos"] = segmentos
            sesion["hablantes"] = hablantes
            return self._send_json(200, {"segmentos": segmentos, "hablantes": hablantes})

        if p.path == "/api/guardar_voz_registro":
            largo = int(self.headers.get("Content-Length", 0) or 0)
            body = json.loads(self.rfile.read(largo).decode("utf-8"))
            sid = body.get("sesion_id", "")
            spk_id = body.get("spk_id", "")
            nombre = (body.get("nombre") or "").strip()
            color = body.get("color", "")

            if not nombre:
                return self._send_text(400, "El nombre de la persona es obligatorio")

            with JOBS_LOCK:
                sesion = SESIONES.get(sid)
            if not sesion:
                return self._send_text(404, "Sesión no encontrada")

            # Buscar los segmentos asignados a este spk_id y extraer el centroide promedio
            X = sesion.get("X")
            segmentos = sesion.get("segmentos", [])
            indices_spk = [i for i, s in enumerate(segmentos) if s.get("hablante") == spk_id]

            if not indices_spk or X is None or len(X) <= max(indices_spk):
                return self._send_text(400, "No se encontraron intervenciones para este hablante")

            # Promediar las características acústicas ponderando por confianza/duración
            X_spk = X[indices_spk]
            vector_voz = np.mean(X_spk, axis=0)

            try:
                perfil = registrar_voz_perfil(nombre, vector_voz, color=color)
            except Exception as e:
                return self._send_text(500, str(e))

            # Actualizar el nombre en la sesión en memoria
            for h in sesion.get("hablantes", []):
                if h["id"] == spk_id:
                    h["nombre"] = nombre
                    h["auto_etiquetado"] = True
                    if color:
                        h["color"] = color

            return self._send_json(200, {"ok": True, "perfil": perfil})

        if p.path == "/api/eliminar_voz_registro":
            largo = int(self.headers.get("Content-Length", 0) or 0)
            body = json.loads(self.rfile.read(largo).decode("utf-8"))
            nombre = (body.get("nombre") or "").strip()
            if not nombre:
                return self._send_text(400, "Nombre requerido")
            ok = eliminar_voz_perfil(nombre)
            return self._send_json(200, {"ok": ok})

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
    print(f"Iniciando Estudio de Audio en http://localhost:{PORT}  (Ctrl+C para cerrar)")
    
    # Sincronización automática de GitHub en segundo plano al arrancar
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
