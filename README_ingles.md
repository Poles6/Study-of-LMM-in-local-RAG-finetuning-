# Study on the use of LLMs in local environments: RAG + Fine-tuning (QLoRA)

## Table of Contents
- [Project Description](#project-description)
- [Architecture & Structure](#architecture--structure)
- [Prerequisites](#prerequisites)
- [Workflow Pipeline](#workflow-pipeline)
  - [Step 1: Original RAG](#step-1---use-the-original-rag)
  - [Step 2: Build the Dataset](#step-2---build-the-dataset)
  - [Step 3: Fine-tuning with QLoRA](#step-3---fine-tuning-with-qlora)
  - [Step 4: Import and Test](#step-4---import-to-ollama-and-test)
- [Dataset Format](#dataset-format-datasetjsonl)
- [Hardware Notes](#hardware-notes)

---

## Project Description

The main objective of this project is to develop a pipeline to perform automated fine-tuning on small-sized Language Models (2-6 billion parameters) using research articles. The goal is to discover if it is possible to teach a model of this size new information, or to get it to correctly answer questions that it previously answered incorrectly or incompletely.

The strategy is to start from a RAG (Retrieval-Augmented Generation) system that we use to answer questions about a given article. The LLM's job in this case is to generate the answers to the provided questions using the article's information, and to autonomously generate additional question-and-answer pairs.

These questions and answers will be used to build a dataset. Once the dataset is sufficiently complete after processing several articles, it will be used to fine-tune the chosen base model. For the fine-tuning process, we use the **QLoRA** strategy to ensure that the process remains compatible with local execution.

Once we have the new model after fine-tuning, we can test it to see if it has genuinely learned the new concepts. It can also be directly compared with the base model before fine-tuning.

---

## Architecture & Structure

| File | Purpose |
|---|---|
| `interfaz_ollama.py` | Original RAG — no changes to its functionality. |
| `dataset_builder.py` | Generates the incremental dataset (New Gradio UI). |
| `finetune_qlora.py` | CLI script that executes fine-tuning using QLoRA. |
| `chat_finetuned.py` | Chat interface (Gradio) to test and compare the fine-tuned model. |
| `dataset.jsonl` | Automatically generated dataset (Q&A). |
| `adapter_out/` | Output directory for the LoRA Adapter resulting from training. |

---

## Prerequisites

Ensure you have installed the required dependencies before starting the fine-tuning process:

```bash
pip install transformers peft trl bitsandbytes datasets accelerate
```

---

## Workflow Pipeline

### Step 1 — Use the original RAG

If you only want to use the RAG system without generating datasets, run:

```bash
python interfaz_ollama.py
```
It works exactly as the original version.

### Step 2 — Build the dataset

To start extracting knowledge from papers and generating the training set:

```bash
python dataset_builder.py
```

This will open an app in your browser. For each article:
1. Upload the research article (PDF or TXT).
2. *(Optional)* Upload a TXT file with predefined questions (one per line).
3. Adjust the "Auto-generated Q&A" slider (0–20).
4. Click **Process and add to dataset**.

Each run **appends examples** to the existing `dataset.jsonl` file. Repeat this process with as many articles and questions as you need. The dataset saves the evidence (text chunks) for each answer, allowing you to audit the quality before spending resources on training.

### Step 3 — Fine-tuning with QLoRA

Start the training process using the generated dataset:

```bash
python finetune_qlora.py --base_model Qwen/Qwen2.5-3B-Instruct
```

**Available arguments:**
- `--base_model`: HuggingFace Model (required, e.g., `qwen3-vl:4b`)
- `--dataset`: Path to JSONL (default: `dataset.jsonl`)
- `--output`: Adapter output directory (default: `adapter_out`)
- `--epochs`: Training epochs (default: `3`)
- `--batch`: Batch size (default: `1`)
- `--grad_accum`: Gradient accumulation (default: `8`)
- `--lr`: Learning rate (default: `2e-4`)
- `--seq_len`: Max token length (default: `512`)
- `--lora_r`: LoRA Rank (default: `16`)
- `--lora_alpha`: LoRA Alpha (default: `32`)

Upon completion, the script automatically generates an `adapter_out/Modelfile` containing the exact instructions to import your newly trained model into Ollama.

> **Recommended Base Models:**
> - `Qwen/Qwen2.5-3B-Instruct` — excellent balance between size and quality.
> - `google/gemma-2b-it`
> - `TinyLlama/TinyLlama-1.1B-Chat-v1.0` — very lightweight, ideal for quick proof-of-concept tests.

### Step 4 — Import to Ollama and test

Once training is complete, integrate the model to use it locally:

```bash
# 1. Pull the base model in Ollama (first time only)
ollama pull qwen2.5-3b-instruct

# 2. Import the trained adapter
ollama create mi_modelo_ft -f adapter_out/Modelfile

# 3. Open the testing chat
python chat_finetuned.py
```

The chat works in isolation (no document retrieval via RAG). This allows you to verify if the model can answer using *only* what it learned during fine-tuning. The interface includes a dropdown to switch between the base model and the fine-tuned model for direct comparison.

---

## Dataset Format (`dataset.jsonl`)

Each line in the file is a JSON object with the following schema:

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
*Note: The `evidence` field is stored exclusively for human auditing purposes and is not ingested during the training process.*

---

## Hardware Notes

| Environment / GPU | Recommendation |
|---|---|
| **GPU with ≥8GB VRAM** | QLoRA (4-bit) works perfectly with 3-7B models. |
| **GPU with 4-6GB VRAM** | Use smaller models (1-3B) and adjust the sequence length: `--seq_len 256`. |
| **CPU Only** | It works but is extremely slow; use it **only** to verify that the code runs without errors. |
| **Google Colab (Free T4)** | Ideal for free training; remember to mount your project in Google Drive. |