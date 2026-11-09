# Estudio de Voz, Recorte, Hablantes por Grupo, Subtítulos y Estilo ML

Aplicación web local (`http://localhost:8765`) organizada en **5 pestañas** para todo el flujo de trabajo con grabaciones de clases y reuniones:

1. **Paso 1 — Video o Audio a MP3 y Recorte de Video/Audio por Minuto y Segundo (`mm:ss`)**:
   - Extrae MP3 desde videos (`.mp4`, `.mkv`, `.mov`, `.avi`) o convierte otros audios a MP3.
   - Permite recortar **Video (`.mp4`)**, **Audio (`.mp3`)** o **ambos** indicando el minuto y segundo de inicio (`Desde`) y fin (`Hasta`).
2. **Paso 2 — Transcripción (Whisper local o Subir de Tactiq / YouTube / Whisper / SRT)**:
   - Transcribe localmente con Whisper (guardando las sesiones en `transcripciones/` para abrirlas al instante sin esperar).
   - O importa transcripciones externas (`.txt`, `.srt`, `.vtt`, `.docx`, `.pdf`) detectando automáticamente si el formato es de **Tactiq**, **YouTube**, **Whisper** o **SRT/VTT**, con opción de recortar por minuto y segundo.
3. **Paso 3 — Hablantes por Grupo, Primer Matcheo Automático, Recorte y Aislamiento de Mi Voz**:
   - **Grupos de Hablantes con nombre** (ej. `"General"`, `"Grupo Martes"`, preparado en estructura para *Materia / Cátedra / Comisión*).
   - **Carga de listado de nombres desde Excel (`.xlsx`), `.csv` o `.txt`**.
   - **Primer Matcheo Automático (1ra Clase)**:
     - Detecta cuando cada hablante se presenta en el audio (*"mi nombre es..."*, *"yo soy..."*) y lo cruza con la lista del Grupo.
     - O sincroniza automáticamente el desfase de segundos con la transcripción de **Tactiq** de esa clase y asigna los nombres de Tactiq a cada huella de voz.
   - **Recorte de Voz y de Transcripción por Minuto y Segundo (`mm:ss`) o por Frases de inicio/fin**, con opción de reiniciar el reloj a `00:00` en el punto de recorte.
   - **Silenciar/excluir hablantes** y **Aislar tu voz limpia** en `MP3`, `WAV 44.1 kHz` o `Dataset ZIP (wavs + metadata.csv)` para entrenamiento de clonación de voz.
4. **Paso 4 — Subtítulos y Sincronización con Video sin audio (Google Drive / YouTube)**:
   - Convierte la transcripción (con etiquetas `[Nombre del hablante]`) a `.SRT` y `.VTT`.
   - Permite previsualizar el video local, sincronizar el desfase de segundos con una frase de referencia, recortar el video por minuto y segundo (`mm:ss`) y unir los subtítulos al `.MP4` con `ffmpeg`.
5. **Pestaña 5 — 🧠 Modelo de ML para Entrenar Mi Estilo de Habla y Convertir Guiones**:
   - Entrena un modelo estilométrico + N-gramas + índice RAG (`estilo_habla_modelo.json`) con tus transcripciones reales.
   - Recibe cualquier guión formal o apunte y lo reescribe con tus muletillas, conectores, voseo, ritmo de oración y vocabulario propio.

---

## 🌟 Deseables a Futuro (Roadmap — No hacer ahora)

### 1. 🗣️🎙️ Clonar Mi Voz y Convertir el Texto en Mi Estilo directamente a Audio con Mi Voz
- **Etapa A — Entrenar / Clonar Mi Voz Acústica**:
  - Tomar el audio limpio de tu voz o el `Dataset ZIP` generado en el **Paso 3** y conectarlo con un motor local de clonación de voz (**XTTS-v2**, **F5-TTS** o **RVC**).
- **Etapa B — De Guión Convertido a Mi Estilo → Audio Hablado con Mi Voz Clonada (Texto-a-Mi-Voz)**:
  - En la **Pestaña 5**, una vez que el modelo de ML convirtió el guión a tu forma de hablar (con tus conectores, ritmo y expresiones reales), agregar el botón **"🔊 Generar Audio con Mi Voz Clonada"** para sintetizar ese texto adaptado directamente en un archivo `.mp3` / `.wav` locutado con tu propio timbre, entonación y pausas naturales.

### 2. 🏫 Jerarquía Completa de Grupos por Materia → Cátedra → Comisión
- Expandir el selector de **Grupos de Hablantes** del Paso 3 para organizar cursos por *Materia*, *Cátedra* y *Comisión* (los campos base ya están preparados en `voces_registradas.json`).
