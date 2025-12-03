📚 EA3. Generación de contenido con IA generativa (Parte 3)

📋 Descripción del Proyecto.

🔮 Desarrollar una aplicación integral de inteligencia artificial generativa
implementando técnicas avanzadas de GAN o modelos de difusión

1.  Diseño de la solución generativa.
    Se realiza la selección de una problemática del ámbito educativo enfocada en la necesidad de crear material didáctico de manera rápida, personalizada y de calidad. La solución propuesta emplea técnicas de Inteligencia Artificial para generar contenido educativo adaptado a distintos temas, niveles de complejidad y perfiles de aprendizaje, optimizando así el proceso de creación y actualización de recursos formativos.

2.  Implementación técnica.
    Para generación visual, utiliza e implementa al menos una de las siguientes arquitecturas: Redes Generativas Adversarias (GAN), DCGAN, StyleGAN, etc.
    Se utilizan el modelo de difusión Stable Diffusion con las siguientes características implementadas:
    Modelo Base: Stable Diffusion
    Modelo de difusión probabilística (DDPM - Denoising Diffusion Probabilistic Models)
    Para la generación textual se utiliza Gemini 2.5 con las siguientes características implementadas:
    Modelo: Gemini 2.5 Flash de Google.
    Prompt Engineering Avanzado.
    Dominio Específico: Educación.
    Para la aplicación se utiliza Gradio con las siguientes características:
    Framework: Gradio
    Interfaz web interactiva
    Diseño responsive con tema personalizado
    Componentes: Chatbot, inputs de texto, dropdown, botones, visualización de imágenes

    Los modelos Stable Diffusion y Gemini son modelos pre-entrenados, no se entrenan desde cero. Las visualizaciones se enfocan en la evaluación de experimentos con diferentes configuraciones

3.  Experimentación y optimización.
    Ejecutamos al menos tres experimentos variando parámetros, arquitectura o datos.
    Se crea un dataset para los ejemplos de experimentación.

4.  Aplicación práctica y demostración.
    Configuración de experimentos: 3.
    Baseline (Configuración Estándar)
    Steps: 35 Guidance: 7.5
    Alta calidad (Mas pasos)
    Creativo (Mayor Guidance)
    Steps: 35 Guidance: 12.0

    Se desarrolla una interfaz de usuario amigable que permita probar el sistema.
    La interfaz del usuario fue desarrollada en Gradio.

5.  Análisis crítico y reflexión ética.

    Identifica posibles sesgos en tu modelo y explica dos estrategias para mitigarlos.
    Se identifican los siguientes sesgos en el modelo
    Detección de Sesgos (4 tipos):
    Sesgo de género
    Sesgo de complejidad
    Sesgo cultural
    Sesgo de representación
    Reflexión de Impacto:
    5 impactos positivos (educación, empleo, comunicación)
    5 impactos negativos potenciales
    Análisis de Riesgos:
    5 riesgos identificados con mitigaciones
    8 recomendaciones de uso responsable

        El impacto no se puede determinar, el futuro de la IA en la educación depende de decisiones humanas; el futuro de la educación no es humanos vs IA, sino de humanos + IA, en donde la IA maneje tareas escalables y repetitivas, y los humanos aporten validación, contexto y conexión emocional.
        La tecnología es una herramienta poderosa, pero la sabiduría para usarla responsablemente es exclusivamente humana. El impacto de la IA en educación, empleo y comunicación no está predeterminado; está en nuestras manos construir el futuro que queremos.

---

## 📘 MANUAL DE INSTALACIÓN

    (Para tu proyecto IA Generativa)

🧩 1. Requisitos previos

    Antes de iniciar, necesitas:

    ✔️ Cuenta de Google

    ✔️ Acceso a Google Drive

    ✔️ Conexión a Internet

    ✔️ Archivo del notebook .ipynb (tu archivo EA3_IA_GENERATIVA_GEMINI.ipynb)

    ✔️ (Opcional) Una API Key si usas algún modelo externo (Gemini, OpenAI, HF, etc.)

🗂️ 2. Apertura del Notebook

    🔹 Paso 1: Abrir Google Colab

    Entra a 👉 https://colab.research.google.com

    Haz clic en 📁 File → Upload Notebook

    Selecciona tu archivo EA3_IA_GENERATIVA_GEMINI.ipynb

⚙️ 3. Configuración del entorno Colab

    🔹 Paso 1: Activar GPU (si tu proyecto lo necesita)

    Ve a: Entorno de ejecución ⚡ → Cambiar tipo de entorno de ejecución

    En Acelerador de hardware, elige:

    GPU (T4 o P100 normalmente)

    🔧 Esto permite entrenar modelos o cargar grandes transformadores.

📦 4. Instalación de librerías necesarias

    Tu notebook hará instalación automática, pero en caso de que debas ejecutarlas manualmente:

    🔹 Ejecuta estas celdas en Colab:

    !pip install transformers
    !pip install gradio
    !pip install accelerate
    !pip install google-generativeai   # Si usas Gemini
    !pip install datasets

    Si tu proyecto usa:

    modelos open-source → Hugging Face Transformers

    interfaz web → Gradio

    datasets → HuggingFace Datasets

🔐 5. Configuración de la API (Solo si tu proyecto usa Gemini u OpenAI)

    🔹 Insertar clave API

    Crea una celda secreta:

    import os
    os.environ["GOOGLE_API_KEY"] = "TU_API_KEY_AQUI"


    🔏 Nunca subas tu notebook con la clave escrita.

🧠 6. Descarga o carga del modelo

    Ejemplos:

    📌 Para cargar un modelo pre-entrenado (GPT-2)
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2")

    📌 Para usar Gemini
    import google.generativeai as genai
    genai.configure(api_key=os.environ["GOOGLE_API_])
    model = genai.GenerativeModel("gemini-pro")

🌐 7. Lanzar la interfaz web (Gradio)

    Ejemplo:

    import gradio as gr

    def generar(texto):
        respuesta = model.generate_content(texto)
        return respuesta.text

    demo = gr.Interface(fn=generar,
                        inputs=gr.Textbox(lines=4,label="Prompt"),
                        outputs="text")

    demo.launch(share=True)


    🔗 IMPORTANTE:
    La opción share=True genera una URL pública temporal mientras Colab esté activado.

    Si cierras Colab → la URL deja de funcionar (esto es normal).

🎉 8. Instalación completada

    Tu entorno ya está listo para:

    ejecutar inferencia

    generar texto

    probar tu modelo

    lanzar la interfaz web

---

## 📙 GUÍA DE USUARIO

    (Cómo usar la aplicación generativa en Gradio / Gemini / Transformers)

🏁 1. Inicio de la aplicación

    Abre tu notebook en Google Colab.

    Ejecuta cada celda en orden (▶️).

    Cuando aparezca la interfaz de Gradio:

    Verás un cuadro de texto para ingresar prompts

    Verás un botón Submit / Generate

💬 2. Uso básico de la interfaz.

    🔹 Campo principal: Prompt

    Escribe aquí lo que deseas generar, por ejemplo:

    “Genera un resumen del siguiente texto…”

    “Escribe una historia de ciencia ficción…”

    “Explica como funciona un modelo transformer…”

▶️ 3. Ejecutar la generación

    Presiona:

    🚀 Generate / Enviar / Submit

    El modelo responderá con texto generado automáticamente.

🎛️ 4. Controles adicionales (si tu app los tiene)

    Esto depende de tu notebook, pero normalmente puedes tener:

    🔧 temperature

    Controla creatividad

    0.2 → respuesta estable

    0.8 → respuesta creativa

    🔧 max_length

    Máximo de tokens generados

    🔧 top_p

    Filtro de probabilidad acumulada

    🔧 model_selector

    Elegir entre GPT-2, GPT-Neo, Gemini, etc.

📤 5. Exportar o copiar resultados

    Puedes copiar el texto directamente.

    Puedes pegarlo en Word, Drive u otros sistemas.

    Si tu app tiene botón de descarga:

    Haz clic en ⬇️ Download.

🌐 6. Acceso mediante enlace público

    Cuando aparece una salida así:

    Running on public URL: https://xxxx.gradio.live


    Eso significa:

    ✔️ Cualquier persona con el enlace puede usar la app

    ❌ Deja de funcionar cuando Colab se desconecta

    ✔️ Se puede reactivar ejecutando de nuevo la celda

    🛠️ 7. Resolución de problemas comunes

    ⚠️ “ModuleNotFoundError”

    → Ejecuta de nuevo la celda de instalación de librerías.

    ⚠️ “API key no válida”

    → Verifica tu clave.
    → No incluyas espacios ni saltos de línea.

    ⚠️ “La URL pública dejó de funcionar”

    → Es normal cuando Colab se apaga. Ejecuta nuevamente la celda demo.launch().

    ⚠️ El modelo tarda mucho

    → Activa GPU en Colab.
    → Reduce “max_length”.

🧭 8. Flujo típico de uso

    Abrir Colab

    Ejecutar celdas de instalación

    Configurar API Key (si aplica)

    Cargar modelo

    Ejecutar interfaz Gradio

    Usar la URL

    Generar texto

    Exportar resultados

⭐ 9. Buenas prácticas

    Usa prompts claros

    No incluyas datos sensibles

    Guarda tus prompts en un documento

    Si haces fine-tuning, documenta tu dataset

    Activa GPU para entrenar modelos
