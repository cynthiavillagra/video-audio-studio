# 🎙️ Estudio de Audio, Hablantes por Grupo, Subtítulos y Estilo ML

Aplicación web **100% local y privada** (`http://localhost:8765`) que reúne en un solo lugar todo el flujo de trabajo para grabaciones, reuniones, edición de voz, subtitulado de videos mudos y adaptación de guiones a tu propio estilo de hablar.

---

## 👋 ¿Qué hace esta aplicación? (Resumen para quien ingresa por primera vez)

No hace falta seguir siempre todos los pasos en orden: **podés entrar directamente a la pestaña que necesites según lo que tengas a mano** (un video crudo, un audio MP3, una transcripción de Tactiq/YouTube o un guión escrito).

| Pestaña | ¿Para qué sirve? | ¿Qué podés hacer ahí? |
| :--- | :--- | :--- |
| **1. 🎵 Video / Audio a MP3 y Recorte** | Convertir formatos y recortar por tiempo | • Subir cualquier video (`.mp4`, `.mkv`, `.mov`, `.avi`) y extraer su audio en `.mp3`.<br>• Convertir cualquier audio (`.wav`, `.m4a`, `.ogg`) a `.mp3`.<br>• **Recortar Video (`.mp4`) o Audio (`.mp3`)** indicando el **minuto y segundo exacto** de inicio (`Desde`) y fin (`Hasta`).<br>• **Dividir en partes (split)** un video y/o audio: en **varios puntos de corte** (`10:00, 25:30, 1:02:15`) o en **partes de duración fija** (cada `10:00`), con vista previa de las partes y descarga de todas en un `.ZIP`. |
| **2. 📝 Transcripción** | Crear o importar una transcripción | • **Opción A (Con IA local)**: Transcribir un MP3 con Whisper y separar las voces de los hablantes (las sesiones quedan guardadas en disco para abrirlas al instante sin volver a esperar).<br>• **Opción B (Subir transcripción ya hecha)**: Subir un archivo `.txt`, `.srt`, `.vtt`, `.docx` o `.pdf`. **Detecta automáticamente si es de Tactiq, YouTube, Whisper o SRT**, y permite recortarla por minuto y segundo. |
| **3. 👥 Hablantes, Mi Voz y Recorte** | Etiquetar voces por Grupo, silenciar hablantes y aislar tu voz | • **Grupos de Hablantes**: organizar nombres y huellas de voz por grupo (y subir listado de nombres desde un **Excel `.xlsx`**, `.csv` o `.txt`).<br>• **Primer Matcheo Automático (Presentaciones / Tactiq)**: etiquetar las voces automáticamente buscando cuando cada hablante dice su nombre (*"mi nombre es..."*, *"yo soy..."*) o **sincronizando automáticamente con la transcripción de Tactiq**.<br>• **Diagnóstico claro**: ver cuántos hablantes se separaron (con botones de 1 clic para reagrupar en 2 a 6 hablantes) y si se detectó tu voz.<br>• **Recortar Voz y Transcripción** diciendo el **minuto y segundo (`mm:ss`)** o la **frase de inicio y fin**.<br>• **Silenciar/excluir hablantes** o marcar **`⭐ Esta es mi voz`** para exportar solo tu voz limpia en `MP3`, `WAV` o `Dataset ZIP`. |
| **4. 🎬 Subtítulos y Video (Drive)** | Salvar videos grabados sin audio y subtitular | • Ideal para el caso *"grabé un video en Drive, se transcribió bien pero no salió el audio"*.<br>• Convertís la transcripción (con el nombre de cada hablante) en subtítulos **`.SRT`** o **`.VTT`**.<br>• Previsualizás el video local con los subtítulos superpuestos en vivo, sincronizás el desfase de segundos con una frase de referencia, recortás el video por minuto:segundo o unís los subtítulos en un nuevo `.MP4`. |
| **6. 🔇 Video para Publicar** | Publicar una clase sin las voces de los alumnos | • Subís un video (o elegís uno de `audios/`), reconoce las voces (o reutiliza un análisis guardado) y elegís **qué voces silenciar**.<br>• Lo que dicen las voces silenciadas queda como **subtítulo** con nombre **anonimizado** (`Alumno 1`, `Alumno 2`…), **real** o **sin nombre**; tu voz se escucha normal y no se subtitula.<br>• Opción de **nivelar todo el audio** y margen de seguridad de 0,2 s.<br>• Sale un `.MP4` con la pista de subtítulos interna + un `.SRT` aparte, con vista previa en la app. |
| **5. 🧠 Mi Estilo (ML) y Guiones** | Entrenar un modelo con tu forma de hablar y adaptar guiones | • **Entrenar Modelo ML**: analiza tus transcripciones reales y aprende tus muletillas, conectores (*"vamos a ver"*, *"o sea"*, *"fíjense que"*, *"por ejemplo"*, *"acá"*), ritmo de frase y vocabulario.<br>• **Convertir Guión a Mi Estilo**: le pegás cualquier guión formal o texto neutro y lo reescribe exactamente con tu forma natural de expresarte, mostrando qué fragmentos reales tuyos tomó de referencia (RAG). |



### 🎬 Ejemplo: una clase grabada con OBS

1. **Video / Audio**: recortás el video (por ejemplo, el principio y el final que sobran) y extraés el audio en MP3.
2. **Transcripción**: con ese MP3 hacés la transcripción con Whisper, que separa a los hablantes. Si la clase la transcribiste con Tactiq, subís esa: ya viene con los nombres de cada uno. Si tiene partes que quedaron fuera del video recortado, la recortás al subirla.
3. **Hablantes y mi voz**: etiquetás quién es quién y guardás las voces en el grupo, así en las próximas clases se reconocen solas.
4. **Subtítulos y video**: generás los subtítulos (SRT) para subirlos junto al video, por ejemplo en Drive. Si el video salió sin audio por error, ahí sincronizás la transcripción con la imagen para que no quede desfasada.
5. **Video para publicar** *(opcional)*: si vas a compartir la clase fuera del curso, silenciás las voces de los alumnos y lo que dijeron queda como subtítulo.

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
