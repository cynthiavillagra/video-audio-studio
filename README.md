# 🎙️ Estudio de Audio, Hablantes por Grupo, Subtítulos y Estilo ML

Aplicación web **100% local y privada** (`http://localhost:8765`) que reúne en un solo lugar todo el flujo de trabajo para clases grabadas, reuniones, edición de voz, subtitulado de videos mudos y adaptación de guiones a tu propio estilo de hablar.

---

## 👋 ¿Qué hace esta aplicación? (Resumen para quien ingresa por primera vez)

No hace falta seguir siempre todos los pasos en orden: **podés entrar directamente a la pestaña que necesites según lo que tengas a mano** (un video crudo, un audio MP3, una transcripción de Tactiq/YouTube o un guión escrito).

| Pestaña | ¿Para qué sirve? | ¿Qué podés hacer ahí? |
| :--- | :--- | :--- |
| **1. 🎵 Video / Audio a MP3 y Recorte** | Convertir formatos y recortar por tiempo | • Subir cualquier video (`.mp4`, `.mkv`, `.mov`, `.avi`) y extraer su audio en `.mp3`.<br>• Convertir cualquier audio (`.wav`, `.m4a`, `.ogg`) a `.mp3`.<br>• **Recortar Video (`.mp4`) o Audio (`.mp3`)** indicando el **minuto y segundo exacto** de inicio (`Desde`) y fin (`Hasta`). |
| **2. 📝 Transcripción** | Crear o importar una transcripción | • **Opción A (Con IA local)**: Transcribir un MP3 con Whisper y separar las voces de los hablantes (las sesiones quedan guardadas en disco para abrirlas al instante sin volver a esperar).<br>• **Opción B (Subir transcripción ya hecha)**: Subir un archivo `.txt`, `.srt`, `.vtt`, `.docx` o `.pdf`. **Detecta automáticamente si es de Tactiq, YouTube, Whisper o SRT**, y permite recortarla por minuto y segundo. |
| **3. 👥 Hablantes, Mi Voz y Recorte** | Etiquetar voces por Grupo, silenciar hablantes y aislar tu voz | • **Grupos de Hablantes**: organizar nombres y huellas de voz por grupo (y subir listado de nombres desde un **Excel `.xlsx`**, `.csv` o `.txt`).<br>• **Primer Matcheo Automático (1ra Clase)**: etiquetar las voces automáticamente buscando cuando cada hablante dice su nombre (*"mi nombre es..."*, *"yo soy..."*) o **sincronizando automáticamente con la transcripción de Tactiq**.<br>• **Diagnóstico claro**: ver cuántos hablantes se separaron (con botones de 1 clic para reagrupar en 2 a 6 hablantes) y si se detectó tu voz.<br>• **Recortar Voz y Transcripción** diciendo el **minuto y segundo (`mm:ss`)** o la **frase de inicio y fin**.<br>• **Silenciar/excluir hablantes** o marcar **`⭐ Esta es mi voz`** para exportar solo tu voz limpia en `MP3`, `WAV` o `Dataset ZIP`. |
| **4. 🎬 Subtítulos y Video (Drive)** | Salvar videos grabados sin audio y subtitular | • Ideal para el caso *"grabé un video en Drive, se transcribió bien pero no salió el audio"*.<br>• Convertís la transcripción (con el nombre de cada hablante) en subtítulos **`.SRT`** o **`.VTT`**.<br>• Previsualizás el video local con los subtítulos superpuestos en vivo, sincronizás el desfase de segundos con una frase de referencia, recortás el video por minuto:segundo o unís los subtítulos en un nuevo `.MP4`. |
| **5. 🧠 Mi Estilo (ML) y Guiones** | Entrenar un modelo con tu forma de hablar y adaptar guiones | • **Entrenar Modelo ML**: analiza tus transcripciones reales y aprende tus muletillas, conectores (*"vamos a ver"*, *"o sea"*, *"fíjense que"*, *"por ejemplo"*, *"acá"*), ritmo de frase y vocabulario.<br>• **Convertir Guión a Mi Estilo**: le pegás cualquier guión formal o apunte y lo reescribe exactamente como lo explicarías vos en clase, mostrando qué fragmentos reales tuyos tomó de referencia (RAG). |

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
2. **🏫 Jerarquía Completa de Grupos por Materia → Cátedra → Comisión**:
   - Expandir los **Grupos de Hablantes** del Paso 3 para filtrar por *Materia*, *Cátedra* y *Comisión* (estructura ya contemplada en `voces_registradas.json`).
