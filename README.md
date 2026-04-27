# Estudio del uso de LLMs en entornos locales: RAG + Fine-tuning (QLoRA)

## Índice
- [Descripción del Proyecto](#descripción-del-proyecto)
- [Arquitectura y Estructura](#arquitectura-y-estructura)
- [Requisitos Previos](#requisitos-previos)
- [Flujo de Trabajo (Pipeline)](#flujo-de-trabajo-pipeline)
  - [Paso 1: RAG Original](#paso-1---usar-el-rag-original)
  - [Paso 2: Construir el Dataset](#paso-2---construir-el-dataset)
  - [Paso 3: Fine-tuning con QLoRA](#paso-3---fine-tuning-con-qlora)
  - [Paso 4: Importar y Testear](#paso-4---importar-en-ollama-y-testear)
- [Formato del Dataset](#formato-del-dataset-datasetjsonl)
- [Notas sobre Hardware](#notas-sobre-hardware)

---

## Descripción del Proyecto

El principal objetivo de este proyecto es desarrollar un flujo para realizar fine-tuning con modelos de lenguaje de tamaño pequeño (2-6 billones de parámetros) de forma automatizada sobre artículos de investigación. El objetivo es descubrir si es posible enseñar a un modelo de este tamaño información nueva, o conseguir que responda correctamente a preguntas que antes contestaba incorrectamente o de forma incompleta.

La estrategia es partir de un RAG (Retrieval-Augmented Generation) que usamos para responder preguntas sobre un artículo. El trabajo del LLM en este caso es generar las respuestas a las preguntas dadas usando la información del artículo y generar más preguntas y respuestas de forma individual.

Estas preguntas y respuestas se usarán para construir un dataset. Una vez el dataset esté lo suficientemente completo después de haber procesado varios artículos, se utilizará para hacer fine-tuning con el modelo base elegido. Para el fine-tuning aplicamos la estrategia de **QLoRA** para garantizar que el proceso siga siendo compatible para su ejecución en entornos locales.

Una vez tengamos el nuevo modelo tras el fine-tuning, podremos testearlo para comprobar empíricamente su aprendizaje, permitiendo además su comparación directa con el modelo original (antes del fine-tuning).

---

## Arquitectura y Estructura

| Archivo | Función |
|---|---|
| `interfaz_ollama.py` | RAG original — sin cambios en su funcionamiento. |
| `dataset_builder.py` | Genera el dataset incremental mediante una interfaz Gradio. |
| `finetune_qlora.py` | Script CLI que ejecuta el fine-tuning aplicando QLoRA. |
| `chat_finetuned.py` | Interfaz de chat (Gradio) para testear y comparar el modelo entrenado. |
| `dataset.jsonl` | Dataset generado automáticamente (Q&A). |
| `adapter_out/` | Directorio destino para el Adapter LoRA resultante del entrenamiento. |

---

## Requisitos Previos

Asegúrate de instalar las dependencias necesarias antes de comenzar con el proceso de fine-tuning:

```bash
pip install transformers peft trl bitsandbytes datasets accelerate
```

---

## Flujo de Trabajo (Pipeline)

### Paso 1 — Usar el RAG original

Si solo deseas usar el sistema RAG sin generar datasets, ejecuta:

```bash
python interfaz_ollama.py
```
Funciona exactamente igual que la versión original.

### Paso 2 — Construir el dataset

Para comenzar a extraer el conocimiento de los papers y generar el set de entrenamiento:

```bash
python dataset_builder.py
```

Esto abrirá una aplicación en tu navegador. Para cada artículo:
1. Sube el PDF o TXT del artículo de investigación.
2. *(Opcional)* Sube un archivo TXT con preguntas predefinidas (una por línea).
3. Ajusta el slider de "Q&A auto-generados" (0–20).
4. Pulsa **Procesar y añadir al dataset**.

Cada ejecución **añade ejemplos** al archivo `dataset.jsonl` existente. Puedes repetir este proceso con tantos artículos como necesites. El dataset guarda la evidencia (el fragmento de texto usado) para cada respuesta, lo que te permite auditar la calidad antes de gastar recursos en entrenar.

### Paso 3 — Fine-tuning con QLoRA

Inicia el proceso de entrenamiento utilizando el dataset generado:

```bash
python finetune_qlora.py --base_model Qwen/Qwen2.5-3B-Instruct
```

**Argumentos disponibles:**
- `--base_model`: Modelo de HuggingFace (obligatorio, ej: `qwen3-vl:4b`)
- `--dataset`: Ruta al JSONL (default: `dataset.jsonl`)
- `--output`: Directorio para guardar el adapter (default: `adapter_out`)
- `--epochs`: Épocas de entrenamiento (default: `3`)
- `--batch`: Batch size (default: `1`)
- `--grad_accum`: Gradient accumulation (default: `8`)
- `--lr`: Learning rate (default: `2e-4`)
- `--seq_len`: Longitud máxima de tokens (default: `512`)
- `--lora_r`: Rango LoRA (default: `16`)
- `--lora_alpha`: Alpha LoRA (default: `32`)

Al terminar, el script generará automáticamente un archivo `adapter_out/Modelfile` con las instrucciones exactas para importar tu nuevo modelo entrenado en Ollama.

> **Modelos base recomendados:**
> - `Qwen/Qwen2.5-3B-Instruct` — excelente equilibrio entre tamaño y calidad.
> - `google/gemma-2b-it`
> - `TinyLlama/TinyLlama-1.1B-Chat-v1.0` — muy ligero, ideal para pruebas rápidas de concepto.

### Paso 4 — Importar en Ollama y testear

Una vez finalizado el entrenamiento, integra el modelo para usarlo localmente:

```bash
# 1. Descarga el modelo base en Ollama (solo la primera vez)
ollama pull qwen2.5-3b-instruct

# 2. Importa el adapter entrenado
ollama create mi_modelo_ft -f adapter_out/Modelfile

# 3. Abre el chat de pruebas
python chat_finetuned.py
```

El chat funcionará de forma aislada (sin buscar documentos por RAG). De esta manera comprobarás si el modelo es capaz de responder utilizando *solo* lo que aprendió durante el fine-tuning. La interfaz permite cambiar entre el modelo base y el fine-tuned para una comparación directa.

---

## Formato del Dataset (`dataset.jsonl`)

Cada línea del archivo es un objeto JSON con la siguiente estructura:

```json
{
  "id": "paper1_q003_a1b2",
  "article_id": "paper1",
  "question": "¿Qué máquina se usó para generar el haz?",
  "answer": "Se usaron dos aceleradores lineales: Kinetron y Oriatron 6e.",
  "evidence": [
    {"chunk_id": "c07", "text": "…Kinetron…4.5 MeV…", "score": 0.78},
    {"chunk_id": "c09", "text": "…Oriatron 6e…6 MeV…", "score": 0.74}
  ],
  "rag": {
    "retriever_top_k": 4,
    "model_teacher": "llama3:latest",
    "temperature": 0.1,
    "prompt_version": "v1"
  }
}
```
*Nota: El campo `evidence` se almacena exclusivamente para propósitos de auditoría humana y no es ingerido durante el proceso de entrenamiento.*

---

## Notas sobre Hardware

| Entorno / GPU | Recomendación |
|---|---|
| **GPU con ≥8GB VRAM** | QLoRA (4-bit) funciona perfectamente con modelos de 3-7B. |
| **GPU con 4-6GB VRAM** | Usa modelos más pequeños (1-3B) y ajusta la secuencia: `--seq_len 256`. |
| **Solo CPU** | Funciona pero es extremadamente lento; úsalo **solo** para verificar que el código no tiene errores. |
| **Google Colab (T4 gratis)** | Ideal para entrenar de forma gratuita; recuerda montar tu proyecto en Google Drive. |