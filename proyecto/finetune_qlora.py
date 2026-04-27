"""
finetune_qlora.py — Fine-tuning con QLoRA a partir del dataset generado por dataset_builder.py

Uso básico:
    python finetune_qlora.py --base_model Qwen/Qwen2.5-3B-Instruct

Todos los argumentos:
    --base_model    Modelo HuggingFace base (obligatorio)
    --dataset       Ruta al JSONL  (default: dataset.jsonl)
    --output        Directorio del adapter (default: adapter_out)
    --epochs        Épocas de entrenamiento (default: 3)
    --batch         Batch size por dispositivo (default: 1)
    --grad_accum    Pasos de gradient accumulation (default: 8)
    --lr            Learning rate (default: 2e-4)
    --seq_len       Longitud máxima de secuencia en tokens (default: 512)
    --lora_r        Rango de LoRA (default: 16)
    --lora_alpha    Alpha de LoRA (default: 32)

Modelos recomendados (pequeños, 1-4B):
    Qwen/Qwen2.5-3B-Instruct          <- buen equilibrio tamaño/calidad
    google/gemma-2b-it
    microsoft/Phi-3-mini-4k-instruct
    TinyLlama/TinyLlama-1.1B-Chat-v1.0  <- muy ligero, para pruebas rápidas

Requisitos:
    pip install transformers peft trl bitsandbytes datasets accelerate
    (bitsandbytes solo necesario con GPU para cuantización 4-bit)
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

# ── Configuración de entrenamiento ────────────────────────────────────────────

SYSTEM_PROMPT = (
    "Eres un experto en análisis de artículos científicos. "
    "Responde siempre en español de forma breve y directa. "
    "Si la información no está disponible, di: No se indica en el documento."
)

# Módulos LoRA que funcionan para la mayoría de arquitecturas (Llama, Qwen, Gemma, Mistral).
# Para Phi-3 puede necesitarse: ["qkv_proj", "o_proj", "gate_up_proj", "down_proj"]
DEFAULT_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]


# ── Carga y limpieza del dataset ──────────────────────────────────────────────

def load_jsonl(path: str):
    records, skipped = [], 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                q = obj.get("question", "").strip()
                a = obj.get("answer", "").strip()
                # Filtramos ejemplos vacíos o con errores del RAG
                if q and a and not a.startswith("[Error"):
                    records.append({"question": q, "answer": a})
                else:
                    skipped += 1
            except json.JSONDecodeError:
                skipped += 1

    print(f"  Ejemplos válidos: {len(records)}")
    if skipped:
        print(f"  Ejemplos descartados (vacíos o con error): {skipped}")
    return records


def to_messages_format(records):
    """Convierte cada par Q&A al formato messages estándar para SFT."""
    return [
        {
            "messages": [
                {"role": "system",    "content": SYSTEM_PROMPT},
                {"role": "user",      "content": r["question"]},
                {"role": "assistant", "content": r["answer"]},
            ]
        }
        for r in records
    ]


# ── Entrenamiento ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="QLoRA fine-tuning a partir del dataset del RAG",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--base_model",  required=True,
                        help="ID del modelo HuggingFace base, ej: Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--dataset",     default="dataset.jsonl")
    parser.add_argument("--output",      default="adapter_out")
    parser.add_argument("--epochs",      type=int,   default=3)
    parser.add_argument("--batch",       type=int,   default=1)
    parser.add_argument("--grad_accum",  type=int,   default=8)
    parser.add_argument("--lr",          type=float, default=2e-4)
    parser.add_argument("--seq_len",     type=int,   default=512)
    parser.add_argument("--lora_r",      type=int,   default=16)
    parser.add_argument("--lora_alpha",  type=int,   default=32)
    args = parser.parse_args()

    print("\n=== Fine-tuning con QLoRA ===")
    print(f"  Modelo base : {args.base_model}")
    print(f"  Dataset     : {args.dataset}")
    print(f"  Output      : {args.output}")
    print(f"  GPU         : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  VRAM        : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ── 1. Dataset ─────────────────────────────────────────────────────────────
    print("\n[1/5] Cargando dataset...")
    if not Path(args.dataset).exists():
        print(f"ERROR: no se encuentra {args.dataset}. Genera ejemplos primero con dataset_builder.py")
        sys.exit(1)

    records = load_jsonl(args.dataset)
    if not records:
        print("ERROR: el dataset no contiene ejemplos válidos.")
        sys.exit(1)

    # ── 2. Tokenizer ───────────────────────────────────────────────────────────
    print("\n[2/5] Cargando tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Formatear a texto usando el chat template del modelo
    raw_data = to_messages_format(records)
    formatted = []
    for item in raw_data:
        try:
            text = tokenizer.apply_chat_template(
                item["messages"],
                tokenize=False,
                add_generation_prompt=False,
            )
            formatted.append({"text": text})
        except Exception as e:
            print(f"  [Aviso] No se pudo aplicar chat template: {e}. Usando formato manual.")
            # Fallback manual si el modelo no tiene chat template definido
            msgs = item["messages"]
            text = (
                f"<|system|>\n{msgs[0]['content']}\n"
                f"<|user|>\n{msgs[1]['content']}\n"
                f"<|assistant|>\n{msgs[2]['content']}"
            )
            formatted.append({"text": text})

    dataset = Dataset.from_list(formatted)
    print(f"  Ejemplos para entrenamiento: {len(dataset)}")

    # ── 3. Modelo ──────────────────────────────────────────────────────────────
    print("\n[3/5] Cargando modelo...")
    use_gpu = torch.cuda.is_available()

    if use_gpu:
        # QLoRA: 4-bit NF4 con double quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            quantization_config=bnb_config,
            device_map="auto",
        )
        print("  Modo: QLoRA (4-bit NF4)")
    else:
        # CPU: sin cuantización, válido para experimentos pequeños
        print("  Sin GPU detectada — cargando en CPU (float32). Será lento.")
        print("  Recomendación: usa Google Colab (GPU gratuita) para entrenamientos reales.")
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            torch_dtype=torch.float32,
            device_map="cpu",
        )
        print("  Modo: LoRA en CPU (sin cuantización)")

    # ── 4. Configuración LoRA ─────────────────────────────────────────────────
    print("\n[4/5] Configurando LoRA...")
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=DEFAULT_TARGET_MODULES,
        # Nota: si el modelo es Phi-3 u otra arquitectura no Llama,
        # puede que necesites ajustar target_modules manualmente.
    )
    print(f"  r={args.lora_r}, alpha={args.lora_alpha}, módulos={DEFAULT_TARGET_MODULES}")

    # ── 5. Entrenamiento ───────────────────────────────────────────────────────
    print("\n[5/5] Entrenando...")
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    sft_config = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        max_seq_length=args.seq_len,
        dataset_text_field="text",
        logging_steps=10,
        save_steps=200,
        fp16=False,
        bf16=use_gpu,
        report_to="none",        # sin wandb/tensorboard por defecto
        optim="paged_adamw_8bit" if use_gpu else "adamw_torch",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        peft_config=peft_config,
        args=sft_config,
    )

    trainer.train()

    # ── Guardar ────────────────────────────────────────────────────────────────
    trainer.model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"\n✓ Adapter guardado en: {output_dir.resolve()}/")

    # ── Generar Modelfile para Ollama ──────────────────────────────────────────
    base_name      = args.base_model.split("/")[-1].lower()
    modelfile_path = output_dir / "Modelfile"
    modelfile_path.write_text(
        f"# Importa el adapter fine-tuned en Ollama.\n"
        f"# IMPORTANTE: el modelo base en Ollama debe ser exactamente el mismo\n"
        f"# que usaste en --base_model ({args.base_model}).\n\n"
        f"FROM {base_name}\n"
        f"ADAPTER {output_dir.resolve()}\n"
        f'SYSTEM "{SYSTEM_PROMPT}"\n',
        encoding="utf-8",
    )

    print(f"""
╔══════════════════════════════════════════════════════════════╗
║              Pasos para usar el modelo en Ollama             ║
╚══════════════════════════════════════════════════════════════╝

1. Asegúrate de tener el modelo base descargado en Ollama:
   ollama pull {base_name}

2. Crea el modelo con el adapter:
   ollama create mi_modelo_ft -f {modelfile_path.resolve()}

3. Prueba el modelo con el chat:
   python chat_finetuned.py
   (selecciona "mi_modelo_ft" en el desplegable)

4. O prueba directamente en terminal:
   ollama run mi_modelo_ft
""")


if __name__ == "__main__":
    main()