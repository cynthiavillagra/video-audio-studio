# Estudio de Voz: Aislar Mi Voz (Entrenar Modelos), Auto-etiquetar Hablantes, Recortar por Frases y Transcribir

Aplicación web local (`http://localhost:8765`) pensada para:
1. **Aislar solo tu voz** (limpia, sin silencios ni ruidos) para entrenar modelos de voz (XTTS, RVC, ElevenLabs, etc.).
2. **Registro de Voces Conocidas (Auto-etiquetado automático)**: guardar perfiles de voces en una clase o reunión para que en grabaciones futuras reconozca quién habla y le asigne su nombre automáticamente.
3. **Detectar marcas de tiempo por frase de inicio y fin** para recortar tramos exactos.
4. **Sincronizador automático de GitHub**: detecta al iniciar si hay cambios nuevos en GitHub hechos desde otra PC y hace `pull` automático (o sube cambios con `push`).
5. **Transcribir y generar subtítulos** con marcas de tiempo interactivas.

Todo corre 100% de manera local y privada en tu PC.

## Requisitos

- **Python 3.10+** con `openai-whisper` (o `faster-whisper`), `numpy` y `scipy`.
- **ffmpeg** en el PATH (`ffmpeg -version`).
- **git** instalado en el sistema.

## Uso paso a paso

### 1. Iniciar la aplicación y sincronización
- Hacé doble clic en `Extraer audio.bat` (o ejecutá `python app.py` en la terminal).
- **Sincronización automática de GitHub**:
  - Al abrirse, comprueba automáticamente contra GitHub. Si hiciste cambios o grabaste nuevas voces desde otra PC, hace `git pull` automático para tener todo al día.
  - En la barra superior derecha de la web tenés el botón **`🔄 Sincronizar`** para forzar pull/push en cualquier momento.
  - Si aún no creaste o vinculaste tu repositorio remoto en GitHub, tenés el botón **`⚙️ Configurar`** (o podés hacer `git remote add origin https://github.com/tu-usuario/tu-repo.git`).
  - *.gitignore ya está configurado* para sincronizar el código y el registro de voces (`voces_registradas.json`) protegiendo y omitiendo los audios pesados de la carpeta `audios/`.

---

### 2. Cargar videos o audios (No arranca solo)
- Arrastrá tus archivos (`.mkv`, `.mp4`, `.mov`, `.mp3`, `.wav`, etc.) a la zona de carga o elegí un audio previo de la carpeta `audios/`.
- Podés revisar la lista, quitar archivos y configurar las opciones (calidad MP3, modelo Whisper, mono/estéreo, o recortar por tiempo/frases).
- Hacé clic en **🗣️ Transcribir, detectar hablantes y buscar frases**.

---

### 3. 🧠 Registro de Voces y Auto-etiquetado en futuras grabaciones
- **En la primera clase/reunión donde hablen todos**:
  1. En las tarjetas de hablantes, escuchá los fragmentos con **▶ Ejemplo** para identificar quién es quién.
  2. Escribí el nombre de la persona en la tarjeta (ej. `Kevin`, `Martina`, `Profesor`).
  3. Hacé clic en **`💾 Guardar voz`**: esto calcula la huella acústica (timbre, formantes y tono vocal) y la guarda de forma permanente en `voces_registradas.json`.
- **En las siguientes grabaciones (incluso en otra PC gracias a la sincronización)**:
  - Al procesar una nueva clase, la aplicación compara las voces detectadas contra el registro. Si coincide con una persona guardada, le asigna su nombre automáticamente y le coloca la etiqueta **`🧠 Reconocido`**.
  - En la parte superior tenés el panel desplegable **"Ver voces guardadas"** donde podés revisar cuántas personas tenés registradas y eliminar perfiles si lo deseás.

---

### 4. ✂️ Recortar por frase de inicio y frase de finalización
- Escribí la frase con la que empezás (ej. *"bueno empezamos la clase"*) y la frase de cierre (ej. *"hasta acá llegamos hoy"*).
- Hacé clic en **🔍 Detectar marcas de tiempo**: la app busca las frases en el texto de la transcripción, calcula el segundo exacto y delimita el tramo `[Desde → Hasta]`.
- Podés escuchar el fragmento encontrado con `▶` o usar los botones rápidos **`⛳ Inicio`** y **`🏁 Fin`** en cualquier línea de la transcripción.
- Podés descargar directamente ese tramo con **`✂️ Descargar este recorte directo (MP3)`**.

---

### 5. 🎯 Aislar tu voz para entrenar un modelo de voz
- Hacé clic en **`⭐ Es mi voz (aislar)`** en tu tarjeta de hablante: silencia automáticamente a todos los demás.
- Elegí el modo **`🎯 Aislar SOLO la voz activa`**:
  - Une únicamente tus intervenciones limpias (evitando ruidos o voces no transcritas entre frases).
  - Podés limitar a **"Los mejores 1, 2, 3, 5 o 10 minutos"** y descartar muletillas cortas (`< 1 seg`).
- Formatos de exportación disponibles:
  - **MP3** (listo para ElevenLabs o plataformas web).
  - **WAV 44.1 kHz sin pérdida** (máxima fidelidad).
  - **📦 Dataset ZIP (`wavs/*.wav` + `metadata.csv`)** para modelos locales como XTTS, RVC, Piper o GPT-SoVITS.

---

### 6. Descargar transcripción
- Descargá el texto en **TXT (con tiempos)**, **TXT (texto corrido)** o **Subtítulos SRT**, con opción de incluir o excluir hablantes silenciados y mostrar nombres.

---

## 🚀 Próximas funciones anotadas a futuro

### 1. Clonación de Voz con Audio (Voz Acústica) y Entrenamiento de "Forma de Hablar" con Transcripciones (Estilo/LLM)
- **A. Clonación Acústica de Voz (Audio -> Timbre/Tono)**:
  - Usar los audios limpios aislados (`.wav` / dataset `.zip`) para entrenar o ajustar un clon de voz local (ej. **XTTSv2**, **F5-TTS**, **Fish-Speech**, **ChatTTS** o **RVC**).
  - Generar síntesis de voz (Text-to-Speech) que hable con tu timbre exacto a partir de cualquier texto nuevo.
- **B. Entrenamiento de "Forma de Hablar" (Texto -> Personalidad/Vocabulario)**:
  - Tomar las transcripciones limpias acumuladas de tus clases o charlas (exportadas en `.txt` o `.jsonl`).
  - Preparar un dataset de estilo/diálogo para:
    - Hacer **fine-tuning** de un modelo de lenguaje chico (ej. Llama-3-8B / Mistral / Gemma) con LoRA/Unsloth.
    - O armar un **System Prompt + RAG de estilo** que imite tus muletillas típicas, expresiones, forma de explicar temas, ejemplos frecuentes y tono pedagógico.
- **C. Unión de ambos mundos (Avatar/Clon Completo)**:
  - El modelo de texto redacta la respuesta usando tu forma de hablar -> el modelo de clonación lee esa respuesta con tu voz real.

### 2. Subtítulos para YouTube desde Tactiq / TXT / DOCX / PDF
- Parser inteligente multiformato para marcas de tiempo (`00:01:23 Nombre: ...`, `[12:34]`, timestamps superiores).
- Exportación directa en `.srt` y `.vtt` listo para subir a YouTube.

### 3. Sincronización de Video y Subtítulos Externos
- Corrección de desfasajes temporales entre la grabación de OBS y la transcripción de Meet/Tactiq (+/- segundos o sincronización por frase ancla).

### 4. Unir Video y Subtítulos (Salvar clases silenciadas)
- Incrustar/quemar subtítulos sobre el video o empaquetar pista `.srt` en el MP4/MKV para rescatar clases donde el micrófono no grabó o no se capturaron las voces de los alumnos.
