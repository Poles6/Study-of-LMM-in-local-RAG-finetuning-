"""
finetune_qlora.py — Fine-tuning con QLoRA a partir del dataset generado por dataset_builder.py

Uso básico:
    python finetune_qlora.py --base_model Qwen/Qwen2.5-3B-Instruct

⚠️ IMPORTANTE: --base_model espera un ID de HuggingFace (org/modelo), NO un tag de Ollama.
Para que luego puedas importar el adapter en Ollama, elige un modelo que exista en
AMBAS plataformas. Estas son combinaciones verificadas:

    HuggingFace ID                       | Tag Ollama         | Tamaño | Notas
    -------------------------------------|--------------------|--------|---------------------------
    TinyLlama/TinyLlama-1.1B-Chat-v1.0   | tinyllama:latest   | 1.1B   | CPU OK para pruebas rápidas
    Qwen/Qwen2.5-3B-Instruct             | qwen2.5:3b         | 3B     | Recomendado, necesita GPU
    google/gemma-2-2b-it                 | gemma2:2b          | 2B     | Buen equilibrio
    mistralai/Mathstral-7B-v0.1          | mathstral:latest   | 7B     | Necesita ≥10GB VRAM

Argumentos:
    --base_model    HF model ID (obligatorio)
    --dataset       Ruta al JSONL  (default: dataset.jsonl)
    --output        Directorio del adapter (default: adapter_out)
    --epochs        Épocas de entrenamiento (default: 3)
    --batch         Batch size por dispositivo (default: 1)
    --grad_accum    Pasos de gradient accumulation (default: 8)
    --lr            Learning rate (default: 2e-4)
    --seq_len       Longitud máxima de secuencia en tokens (default: 512)
    --lora_r        Rango de LoRA (default: 16)
    --lora_alpha    Alpha de LoRA (default: 32)

Requisitos:
    pip install transformers peft trl bitsandbytes datasets accelerate
    (bitsandbytes solo necesario con GPU para cuantización 4-bit)
"""

import argparse
import inspect
import json
import sys
from pathlib import Path

# ── Fix de encoding para Windows ──────────────────────────────────────────────
# TRL carga plantillas .jinja con Path.read_text() sin especificar encoding.
# En Windows el codec por defecto es cp1252 y eso provoca UnicodeDecodeError al
# importar `from trl import SFTTrainer`. Parcheamos read_text para usar utf-8.
if sys.platform == "win32":
    import pathlib
    _orig_read_text = pathlib.Path.read_text
    def _patched_read_text(self, encoding=None, errors=None):
        if encoding is None:
            encoding = "utf-8"
        return _orig_read_text(self, encoding=encoding, errors=errors)
    pathlib.Path.read_text = _patched_read_text  # type: ignore

# ── Resto de imports (el orden importa: trl debe importarse DESPUÉS del fix) ──
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

# Mapeo HuggingFace ID → tag de Ollama. Los nombres NO coinciden 1:1 entre las
# dos plataformas, así que es necesario un mapeo explícito. Si usas un modelo
# que no esté aquí, pásalo manualmente con --ollama_base.
HF_TO_OLLAMA = {
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0":  "tinyllama",
    "Qwen/Qwen2.5-3B-Instruct":            "qwen2.5:3b",
    "Qwen/Qwen2.5-1.5B-Instruct":          "qwen2.5:1.5b",
    "Qwen/Qwen2.5-7B-Instruct":            "qwen2.5:7b",
    "google/gemma-2-2b-it":                "gemma2:2b",
    "google/gemma-2-9b-it":                "gemma2:9b",
    "mistralai/Mathstral-7B-v0.1":         "mathstral",
    "microsoft/Phi-3-mini-4k-instruct":    "phi3:mini",
    "meta-llama/Llama-3.2-1B-Instruct":    "llama3.2:1b",
    "meta-llama/Llama-3.2-3B-Instruct":    "llama3.2:3b",
}

# Módulos LoRA que funcionan para la mayoría de arquitecturas (Llama, Qwen, Gemma, Mistral).
# Para Phi-3 puede necesitarse: ["qkv_proj", "o_proj", "gate_up_proj", "down_proj"]
DEFAULT_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]


# ── Validación del nombre del modelo ──────────────────────────────────────────

def validate_base_model(name: str) -> None:
    """Detecta nombres de modelo incorrectos y da feedback útil antes de fallar feo."""
    # Tag de Ollama tiene formato "modelo:tag" sin "/"
    looks_like_ollama = ":" in name and "/" not in name
    # ID de HuggingFace tiene formato "org/modelo"
    looks_like_hf = "/" in name

    if looks_like_ollama or not looks_like_hf:
        print(f"\n❌ '{name}' no parece un ID de HuggingFace válido.\n")
        print("Este script necesita un ID con formato 'org/modelo', no un tag de Ollama.")
        print("Ollama y HuggingFace son sistemas distintos:")
        print("  • Ollama distribuye modelos en formato GGUF (solo inferencia)")
        print("  • HuggingFace los distribuye en PyTorch (necesario para entrenar)\n")
        print("Modelos compatibles con ambas plataformas:")
        print("  HuggingFace ID                       Tag Ollama          Tamaño")
        print("  ─────────────────────────────────── ──────────────────  ──────")
        print("  TinyLlama/TinyLlama-1.1B-Chat-v1.0   tinyllama:latest    1.1B")
        print("  Qwen/Qwen2.5-3B-Instruct             qwen2.5:3b          3B")
        print("  google/gemma-2-2b-it                 gemma2:2b           2B")
        print("  mistralai/Mathstral-7B-v0.1          mathstral:latest    7B\n")
        print("Ejemplo:")
        print("  python finetune_qlora.py --base_model Qwen/Qwen2.5-3B-Instruct\n")
        sys.exit(1)

    # Aviso si el modelo parece multimodal (VL = Vision-Language)
    lower = name.lower()
    if any(kw in lower for kw in ["-vl", "-vision", "llava", "multimodal"]):
        print(f"\n⚠️  AVISO: '{name}' parece un modelo vision-language (multimodal).")
        print("    Este script está pensado para fine-tuning de texto puro.")
        print("    El entrenamiento puede fallar o dar resultados raros.\n")
        respuesta = input("¿Continuar de todas formas? [s/N]: ").strip().lower()
        if respuesta not in ("s", "si", "sí", "y", "yes"):
            sys.exit(0)


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
    parser.add_argument("--ollama_base", default=None,
                        help="Tag de Ollama del modelo base (para el Modelfile). "
                             "Si se omite, se deduce automáticamente del mapeo conocido.")
    args = parser.parse_args()

    # Validar el ID del modelo antes de empezar
    validate_base_model(args.base_model)

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

    # ── Compatibilidad con distintas versiones de TRL ──
    # TRL ≥ 0.16.0 renombró max_seq_length → max_length en SFTConfig
    sft_config_params = inspect.signature(SFTConfig.__init__).parameters
    if "max_length" in sft_config_params:
        seq_len_kwarg = "max_length"
    elif "max_seq_length" in sft_config_params:
        seq_len_kwarg = "max_seq_length"
    else:
        seq_len_kwarg = None
        print("  [Aviso] No se encontró parámetro de longitud máxima en SFTConfig")

    sft_kwargs = dict(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        dataset_text_field="text",
        logging_steps=10,
        save_steps=200,
        fp16=False,
        bf16=use_gpu,
        report_to="none",        # sin wandb/tensorboard por defecto
        optim="paged_adamw_8bit" if use_gpu else "adamw_torch",
    )
    if seq_len_kwarg:
        sft_kwargs[seq_len_kwarg] = args.seq_len

    sft_config = SFTConfig(**sft_kwargs)

    # TRL ≥ 0.16.0 renombró tokenizer → processing_class en SFTTrainer
    sft_trainer_params = inspect.signature(SFTTrainer.__init__).parameters
    if "processing_class" in sft_trainer_params:
        tokenizer_kwarg = "processing_class"
    else:
        tokenizer_kwarg = "tokenizer"

    trainer_kwargs = dict(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        args=sft_config,
    )
    trainer_kwargs[tokenizer_kwarg] = tokenizer

    trainer = SFTTrainer(**trainer_kwargs)

    trainer.train()

    # ── Guardar ────────────────────────────────────────────────────────────────
    trainer.model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"\n✓ Adapter guardado en: {output_dir.resolve()}/")

    # ── Generar Modelfile para Ollama ──────────────────────────────────────────
    # Determinar el tag de Ollama correcto (NO se puede deducir del ID de HF)
    ollama_base = args.ollama_base or HF_TO_OLLAMA.get(args.base_model)
    if not ollama_base:
        print(f"\n⚠️  No conozco el tag de Ollama para '{args.base_model}'.")
        print("    El Modelfile usará un placeholder que debes editar a mano.")
        print("    Para evitar esto en el futuro, pasa el tag con --ollama_base, por ejemplo:")
        print("      python finetune_qlora.py --base_model <HF> --ollama_base tinyllama")
        ollama_base = "<EDITA_ESTO_CON_TU_TAG_DE_OLLAMA>"

    modelfile_path = output_dir / "Modelfile"
    modelfile_path.write_text(
        f"# Importa el adapter fine-tuned en Ollama.\n"
        f"# IMPORTANTE: el modelo base en Ollama debe ser exactamente el mismo\n"
        f"# que usaste en --base_model ({args.base_model}).\n"
        f"#\n"
        f"# 'ADAPTER .' usa ruta relativa (este mismo directorio). Por eso debes\n"
        f"# ejecutar 'ollama create' DESDE este directorio. Las rutas absolutas\n"
        f"# en Windows dan problemas a Ollama (bug conocido con rutas y espacios).\n\n"
        f"FROM {ollama_base}\n"
        f"ADAPTER .\n"
        f'SYSTEM "{SYSTEM_PROMPT}"\n',
        encoding="utf-8",
    )

    print(f"""
╔══════════════════════════════════════════════════════════════╗
║              Pasos para usar el modelo en Ollama             ║
╚══════════════════════════════════════════════════════════════╝

1. Asegúrate de tener el modelo base descargado en Ollama:
   ollama pull {ollama_base}

2. Crea el modelo desde DENTRO del directorio del adapter
   (necesario por un bug de Ollama en Windows con rutas absolutas):

   cd {output_dir}
   ollama create mi_modelo_ft -f .\\Modelfile
   cd ..

3. Prueba el modelo con el chat:
   python chat_finetuned.py
   (selecciona "mi_modelo_ft" en el desplegable)

4. O prueba directamente en terminal:
   ollama run mi_modelo_ft
""")


if __name__ == "__main__":
    main()