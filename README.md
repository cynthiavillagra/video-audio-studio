# 🎙️ Estudio de Audio, Hablantes por Grupo, Subtítulos y Estilo ML

Aplicación web **100% local y privada** (`http://localhost:8765`) que reúne en un solo lugar todo el flujo de trabajo para grabaciones, reuniones, edición de voz, subtitulado de videos mudos y adaptación de guiones a tu propio estilo de hablar.

---

## 👋 ¿Qué hace esta aplicación? (Resumen para quien ingresa por primera vez)

No hace falta seguir siempre todos los pasos en orden: **podés entrar directamente a la pestaña que necesites según lo que tengas a mano** (un video crudo, un audio MP3, una transcripción de Tactiq/YouTube o un guión escrito).

| Pestaña | ¿Para qué sirve? | ¿Qué podés hacer ahí? |
| :--- | :--- | :--- |
| **1. 🎵 Video / Audio a MP3 y Recorte** | Convertir formatos y recortar por tiempo | • Subir cualquier video (`.mp4`, `.mkv`, `.mov`, `.avi`) y extraer su audio en `.mp3`.<br>• Convertir cualquier audio (`.wav`, `.m4a`, `.ogg`) a `.mp3`.<br>• **Recortar Video (`.mp4`) o Audio (`.mp3`)** indicando el **minuto y segundo exacto** de inicio (`Desde`) y fin (`Hasta`).<br>• **Dividir en partes (split)** un video y/o audio: en **varios puntos de corte** (`10:00, 25:30, 1:02:15`) o en **partes de duración fija** (cada `10:00`), con vista previa de las partes y descarga de todas en un `.ZIP`. |
| **2. 📝 Transcripción (crear, vincular y sincronizar)** | Tener la transcripción conectada con su video | • **Crear** con Whisper (de un audio o video, con separación de hablantes) o **subir** una hecha (Tactiq, YouTube, Whisper, SRT; se puede recortar al subirla).<br>• **Vincular con su video o audio** y **sincronizar**: frase de referencia + "Hacer coincidir", desfase ±. Con Whisper queda sincronizada sola. La vinculación y el desfase **quedan guardados** en la transcripción.<br>• Lista de transcripciones guardadas con su estado (sin video / vinculada / sincronizada). |
| **3. 👥 Hablantes** | Saber quién es quién | **1 · ¿Quién es quién?** escuchar cada voz (con el desfase aplicado), poner nombre (se guarda en el grupo), marcar **☆ Soy yo**, completar automáticamente (Tactiq / presentaciones / lista), separar en 2–30 voces.<br>**2 · Revisar la transcripción**: cambiar la persona de una línea, corregir texto, excluir líneas, marcar tramo. **Las correcciones se guardan solas.** |



### 🎬 Ejemplo: una clase grabada con OBS

1. **Video / Audio**: recortás el video (por ejemplo, el principio y el final que sobran) y extraés el audio en MP3.
2. **Transcripción**: con ese MP3 hacés la transcripción con Whisper (separa a los hablantes y queda sincronizada sola). Si la clase la transcribiste con Tactiq, subís esa (ya viene con los nombres), la **vinculás con el video** y la sincronizás. Si tiene partes que quedaron fuera del video recortado, la recortás al subirla.
3. **Hablantes**: etiquetás quién es quién, marcás cuál sos vos y corregís líneas mal asignadas; las voces quedan en el grupo para las próximas clases.
4. **Resultados**: bajás los subtítulos (SRT) para subirlos con el video a Drive. Si el video salió sin audio, usás la receta "Subtítulos de todo el video".
5. **Resultados → Video** *(opcional)*: si vas a compartir la clase fuera del curso, usás "Publicar sin voces de alumnos": se silencian y lo que dijeron queda como subtítulo.

### 📊 Diagrama de Flujo del Workflow (De arriba hacia abajo)

```mermaid
flowchart TD
    IN(["📥 PUNTO DE PARTIDA<br/>¿Qué archivo tenés hoy?"]) --> P1

    subgraph S1 ["PASO 1 · Video / Audio y Recorte por Tiempo"]
        P1["🎵 Paso 1: Cargar Video o Audio<br/>• Extrae el audio a .MP3<br/>• O recorta el Video (.MP4) / Audio (.MP3) diciendo el minuto y segundo (Desde → Hasta)"]
    end

    P1 --> P2A & P2B

    subgraph S2 ["PASO 2 · Obtener la Transcripción"]
        P2A["🗣️ Opción A: Transcribir MP3 acá (Whisper IA)<br/>Transcribe en tu PC y separa las voces de los hablantes"]
        P2B["📄 Opción B: Subir Transcripción ya hecha<br/>Detecta sola si es de Tactiq, YouTube, Whisper o SRT"]
    end

    P2A & P2B --> P3

    subgraph S3 ["PASO 3 · Hablantes por Grupo, Recorte y Mi Voz"]
        P3["👥 Paso 3: Etiquetar Hablantes y Procesar Voz<br/>• Grupos de Hablantes + Listado de nombres desde Excel (.xlsx)<br/>• Auto-Matcheo Inicial: detecta 'mi nombre es...' en el audio o sincroniza con Tactiq<br/>• Silenciar/excluir hablantes y marcar '⭐ Esta es mi voz'<br/>• Recortar Voz y Transcripción por minuto:segundo (mm:ss) o por frase"]
    end

    P3 --> OUT_VOZ["🎙️ Salida de Voz / Audio y Texto<br/>• Tu voz limpia aislada (.MP3 / .WAV / Dataset .ZIP)<br/>• Audio con hablantes silenciados<br/>• Transcripción recortada (.TXT / .SRT)"]
    P3 --> P4
    P3 --> P5

    subgraph S4 ["PASO 4 · Subtítulos y Videos sin Audio (Drive)"]
        P4["🎬 Paso 4: Subtítulos y Sincronización con Video<br/>• Convierte la transcripción etiquetada a .SRT / .VTT<br/>• Sincroniza el desfase de segundos con el video sin audio<br/>• Recorta el video por minuto:segundo o le une los subtítulos en un nuevo .MP4"]
    end

    subgraph S5 ["PESTAÑA 5 · Modelo ML de Mi Estilo y Guiones"]
        P5["🧠 Pestaña 5: Entrenar Mi Estilo (ML) y Convertir Guiones<br/>• Entrena un modelo con tus transcripciones (aprende tus muletillas, conectores y ritmo)<br/>• Recibe cualquier guión neutro y lo reescribe con tu forma real de hablar"]
        P5 -. "Deseable a futuro (Roadmap)" .-> FUT(["🔊 Próxima etapa: Clonar mi voz y convertir el guión en mi estilo directamente a audio con mi voz"])
    end
```

---

## 🔄 Sincronizador Automático con GitHub (Uso en varias PCs)

- Al iniciar la app, verifica automáticamente el estado del repositorio con GitHub:
  - Si actualizaste el código o registraste voces desde otra PC, hace **`git pull` automático**.
  - Si tenés cambios locales nuevos, los comitea y hace **`git push` automático** (incluyendo `voces_registradas.json` y el modelo de estilo, omitiendo los audios pesados).
- En la esquina superior derecha siempre ves el estado de sincronización y el botón **`🔄 Sincronizar`**.

---

## 🚀 Cómo iniciar la aplicación

1. Hacé doble clic en **`Extraer audio.bat`** (o ejecutá `python app.py` en la terminal).
2. Se abrirá automáticamente `http://localhost:8765` en tu navegador.

### Requisitos del sistema
- **Python 3.10+** con `numpy`, `scipy` y `faster-whisper` (o `openai-whisper`).
- **ffmpeg** instalado y disponible en el `PATH`.
- **git** instalado para la sincronización entre PCs.

---

## 🌟 Deseables a Futuro (Roadmap — Próximas etapas)

1. **🗣️🎙️ Clonar Mi Voz y Convertir el Guión en Mi Estilo directamente a Audio con Mi Voz**:
   - **Etapa A (Clonación de Voz)**: Usar los audios limpios y el `Dataset ZIP` exportados en el **Paso 3** para entrenar un modelo local de clonación de voz (**XTTS-v2**, **F5-TTS** o **RVC**).
   - **Etapa B (Texto en Mi Estilo → Audio con Mi Voz)**: Que el guión ya convertido a tu estilo en la **Pestaña 5** tenga un botón para sintetizar el audio `.mp3` / `.wav` hablado directamente con tu propia voz clonada.
2. **🗂️ Jerarquía Multinivel de Grupos de Hablantes (Categoría → Área → Subgrupo)**:
   - Expandir el selector de **Grupos de Hablantes** del Paso 3 para permitir agrupar perfiles de voz en niveles jerárquicos personalizados (por ejemplo *Categoría / Proyecto / Subgrupo*).
