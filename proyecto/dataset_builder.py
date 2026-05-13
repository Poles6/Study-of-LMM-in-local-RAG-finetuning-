"""
dataset_builder.py — Generador incremental de dataset para fine-tuning.

Procesa documentos (PDF/TXT) + preguntas usando el RAG local (Ollama) y
guarda cada par (pregunta, respuesta, evidencia) en dataset.jsonl.

Se puede ejecutar tantas veces como quieras: cada ejecución AÑADE ejemplos
al archivo existente. Cuando el dataset te parezca suficiente, ejecuta
finetune_qlora.py para lanzar el entrenamiento.

Diferencias con interfaz_ollama.py:
  - Procesa las preguntas de una en una para guardar evidencia por chunk.
  - Puede auto-generar preguntas y respuestas adicionales.
  - Guarda el resultado en JSONL auditadle (question + answer + evidence).
"""

import json
import re
import time
import uuid
from pathlib import Path
from typing import List, Dict, Tuple

import gradio as gr
import ollama
import PyPDF2

# ── Configuración ──────────────────────────────────────────────────────────────

DATASET_PATH = Path("dataset.jsonl")
CHUNK_SIZE    = 800   # caracteres máximos por chunk
TOP_K_CHUNKS  = 6     # chunks a pasar al modelo por pregunta
MAX_ANS_TOKENS = 300

SYSTEM_PROMPT = (
    "Eres un experto en análisis de artículos científicos. "
    "Responde siempre en español de forma breve y directa."
)


# ── Lectura de archivos ────────────────────────────────────────────────────────

def _read_document(filepath: str) -> str:
    filepath = str(filepath)
    if filepath.lower().endswith(".pdf"):
        try:
            with open(filepath, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                return "\n".join(p.extract_text() or "" for p in reader.pages).strip()
        except Exception as e:
            return f"[Error leyendo PDF: {e}]"
    for enc in ("utf-8", "latin-1"):
        try:
            with open(filepath, "r", encoding=enc) as f:
                return f.read().strip()
        except UnicodeDecodeError:
            continue
        except Exception as e:
            return f"[Error leyendo archivo: {e}]"
    return "[No se pudo leer el archivo]"


def _read_questions(filepath: str) -> List[str]:
    if not filepath:
        return []
    for enc in ("utf-8", "latin-1"):
        try:
            with open(str(filepath), "r", encoding=enc) as f:
                return [q.strip() for q in f.readlines() if q.strip()]
        except (UnicodeDecodeError, Exception):
            continue
    return []


# ── Chunking y retrieval ligero ───────────────────────────────────────────────

def _chunk_document(text: str) -> List[Dict]:
    """Divide el texto en chunks numerados, respetando frases."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    chunks, current, idx = [], "", 0
    for sent in sentences:
        if len(current) + len(sent) > CHUNK_SIZE and current:
            chunks.append({"chunk_id": f"c{idx:03d}", "text": current.strip()})
            idx += 1
            current = sent
        else:
            current += (" " if current else "") + sent
    if current.strip():
        chunks.append({"chunk_id": f"c{idx:03d}", "text": current.strip()})
    return chunks


def _score(question: str, chunk_text: str) -> float:
    """Jaccard sobre tokens: métrica simple sin dependencias externas."""
    q = set(re.findall(r'\w+', question.lower()))
    c = set(re.findall(r'\w+', chunk_text.lower()))
    if not q or not c:
        return 0.0
    return len(q & c) / len(q | c)


def _retrieve(question: str, chunks: List[Dict], k: int = TOP_K_CHUNKS) -> List[Dict]:
    scored = sorted(chunks, key=lambda ch: _score(question, ch["text"]), reverse=True)
    return [
        {"chunk_id": ch["chunk_id"], "text": ch["text"], "score": round(_score(question, ch["text"]), 4)}
        for ch in scored[:k]
    ]


# ── RAG: pregunta → respuesta + evidencia ─────────────────────────────────────

def _rag_prompt(question: str, evidence: List[Dict]) -> str:
    context = "\n\n".join(f"[{e['chunk_id']}]: {e['text']}" for e in evidence)
    return (
        "Usa únicamente los fragmentos del artículo que se muestran a continuación.\n"
        "Si la respuesta no aparece en los fragmentos, escribe exactamente: "
        '"No se indica en el documento."\n'
        "Responde en 1-3 frases. Al final añade en una línea nueva: "
        '"CITAS: <ids separados por comas>"\n\n'
        f"Fragmentos:\n{context}\n\n"
        f"Pregunta: {question}\n"
    )


def _ask_rag(question: str, chunks: List[Dict], model: str) -> Tuple[str, List[Dict]]:
    evidence = _retrieve(question, chunks)
    prompt = _rag_prompt(question, evidence)
    try:
        resp = ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            options={"temperature": 0.1, "num_predict": MAX_ANS_TOKENS},
        )
        raw = resp.message.content.strip() if hasattr(resp, "message") else str(resp).strip()
    except Exception as e:
        print(f"[RAG error]: {e}")
        return f"[Error: {e}]", evidence

    answer = raw
    # Buscar y extraer la sección de respuesta antes de CITAS
    if "CITAS:" in raw:
        parts = raw.split("CITAS:")
        answer = parts[0].strip()
        # Extraer IDs citados
        cited_str = parts[1].strip() if len(parts) > 1 else ""
        if cited_str and cited_str.lower() != "none":
            cited = {cid.strip() for cid in cited_str.split(",") if cid.strip()}
            filtered = [e for e in evidence if e["chunk_id"] in cited]
            if filtered:
                evidence = filtered
    
    # Validar que tenemos una respuesta real
    if not answer or answer.startswith("[Error"):
        return answer, evidence
    
    return answer, evidence


# ── Auto-generación de Q&A adicionales ───────────────────────────────────────

def _parse_qa_blocks(raw: str, chunk_map: Dict) -> List[Tuple[str, str, List[Dict]]]:
    """Parsea bloques Q/A/CITAS de un texto, robusto a markdown y separadores variables."""
    results = []
    
    # Quitar markdown (** ** y * *)
    text = re.sub(r'\*+', '', raw)
    
    # Dividir el texto por marcadores Q: (con o sin numeración previa)
    # Esto encuentra todas las preguntas, sin depender de "---" como separador
    blocks = re.split(r'(?:^|\n)\s*(?:\d+[\.\)]\s*)?Q\s*[:\.]', text, flags=re.IGNORECASE)
    
    for block in blocks[1:]:  # Saltamos el texto inicial antes del primer Q:
        # Buscar A: dentro del bloque
        a_split = re.split(r'\n\s*A\s*[:\.]', block, maxsplit=1, flags=re.IGNORECASE)
        if len(a_split) < 2:
            continue
        
        question = a_split[0].strip()
        rest = a_split[1]
        
        # Buscar CITAS: opcional
        c_split = re.split(r'\n\s*CITAS?\s*[:\.]', rest, maxsplit=1, flags=re.IGNORECASE)
        answer = c_split[0].strip()
        # Limpiar separadores residuales
        answer = re.sub(r'\n+---+\s*$', '', answer).strip()
        
        cited_str = ""
        if len(c_split) > 1:
            cited_str = c_split[1].split('\n')[0].strip()
            cited_str = re.sub(r'---+\s*$', '', cited_str).strip()
        
        # Validación mínima
        if not question or not answer or len(answer) < 5:
            continue
        
        # Procesar citas
        evidence = []
        if cited_str and cited_str.lower() not in ('none', 'ninguno', 'ninguna', 'n/a', ''):
            cited_ids = [cid.strip() for cid in cited_str.split(",") if cid.strip()]
            evidence = [
                {"chunk_id": cid, "text": chunk_map[cid]["text"], "score": None}
                for cid in cited_ids if cid in chunk_map
            ]
        
        results.append((question, answer, evidence))
    
    return results


def _auto_generate_qa(chunks: List[Dict], model: str, n: int) -> List[Tuple[str, str, List[Dict]]]:
    """Genera n pares Q&A en batches pequeños para mayor fiabilidad."""
    if n <= 0:
        return []
    
    BATCH_SIZE = 5  # Batches pequeños = más fiable que pedir 10 de golpe
    chunk_map = {c["chunk_id"]: c for c in chunks}
    context = "\n\n".join(f"[{c['chunk_id']}]: {c['text']}" for c in chunks[:8])
    
    all_results: List[Tuple[str, str, List[Dict]]] = []
    remaining = n
    batch_idx = 0
    
    while remaining > 0:
        batch_idx += 1
        this_batch = min(BATCH_SIZE, remaining)
        
        # Nota: TODAS las líneas con {variables} llevan prefijo f
        prompt = (
            f"A partir del artículo científico, genera exactamente {this_batch} pares de pregunta y respuesta.\n"
            "Las respuestas deben basarse SOLO en los fragmentos del artículo.\n"
            'Si la respuesta no está en el artículo, escribe: "No se indica en el documento."\n\n'
            "FORMATO OBLIGATORIO (sin numeración, sin negrita, sin markdown):\n"
            "Q: pregunta aquí\n"
            "A: respuesta aquí en 1-2 frases\n"
            "CITAS: chunk_id1, chunk_id2\n\n"
            "Q: otra pregunta\n"
            "A: otra respuesta\n"
            "CITAS: chunk_id3\n\n"
            f"Artículo:\n{context}\n\n"
            f"Genera ahora exactamente {this_batch} pares en el formato indicado:"
        )
        
        try:
            resp = ollama.chat(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                options={"temperature": 0.4, "num_predict": 1200},
            )
            raw = resp.message.content.strip() if hasattr(resp, "message") else ""
        except Exception as e:
            print(f"[Auto-gen batch {batch_idx} error]: {e}")
            break
        
        if not raw:
            print(f"[Auto-gen batch {batch_idx}] respuesta vacía del modelo")
            break
        
        parsed = _parse_qa_blocks(raw, chunk_map)
        print(f"  Batch {batch_idx}: {len(parsed)}/{this_batch} parseadas correctamente")
        
        if len(parsed) == 0:
            # Si no se parseó nada, no insistir (probablemente formato muy raro)
            print(f"  [Aviso] El modelo no respetó el formato. Output recibido:\n{raw[:300]}…")
            break
        
        all_results.extend(parsed)
        remaining -= len(parsed)
    
    print(f"  Total auto-generadas: {len(all_results)}/{n}")
    return all_results


# ── Dataset JSONL ─────────────────────────────────────────────────────────────

def _dataset_stats() -> str:
    if not DATASET_PATH.exists():
        return "📭 Dataset vacío (0 ejemplos)"
    lines  = [l for l in DATASET_PATH.read_text("utf-8").splitlines() if l.strip()]
    arts   = set()
    for line in lines:
        try:
            arts.add(json.loads(line).get("article_id", "?"))
        except Exception:
            pass
    return f"📦 **Dataset actual:** {len(lines)} ejemplos · {len(arts)} artículo(s)"


def _append_entries(entries: List[Dict]) -> None:
    with open(DATASET_PATH, "a", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")


def _clear_dataset() -> str:
    if DATASET_PATH.exists():
        DATASET_PATH.unlink()
    return "🗑️ Dataset borrado."


# ── Proceso principal ─────────────────────────────────────────────────────────

def process_document(doc_file, q_file, model, n_auto, progress=gr.Progress()):
    start_time = time.time()
    
    if not doc_file or not model:
        return "⚠️ Necesitas subir un documento y seleccionar un modelo.", _dataset_stats()

    progress(0.0, desc="Leyendo documento...")
    doc_text = _read_document(doc_file)
    if not doc_text or doc_text.startswith("[Error"):
        return doc_text, _dataset_stats()

    chunks     = _chunk_document(doc_text)
    article_id = Path(str(doc_file)).stem.replace(" ", "_")
    questions  = _read_questions(q_file) if q_file else []

    entries    = []
    log_lines  = [
        f"**Artículo:** `{article_id}` · {len(chunks)} chunks · {len(questions)} preguntas",
        "",
    ]
    total      = len(questions) + (1 if n_auto > 0 else 0)

    # ── Preguntas del archivo ──────────────────────────────────────────────────
    for i, question in enumerate(questions):
        q_start = time.time()
        progress(i / max(total, 1), desc=f"Pregunta {i+1}/{len(questions)}…")
        answer, evidence = _ask_rag(question, chunks, model)
        q_elapsed = time.time() - q_start
        
        entries.append({
            "id":         f"{article_id}_q{i:03d}_{uuid.uuid4().hex[:4]}",
            "article_id": article_id,
            "question":   question,
            "answer":     answer,
            "evidence":   evidence,
            "rag": {
                "retriever_top_k": TOP_K_CHUNKS,
                "model_teacher":   model,
                "temperature":     0.1,
                "prompt_version":  "v1",
            },
        })
        status = "✅" if not answer.startswith("[Error") else "❌"
        q_text = question[:70] + "…" if len(question) > 70 else question
        a_text = answer[:100] + "…" if len(answer) > 100 else answer
        log_lines.append(f"{status} **Q:** {q_text}\n   **A:** {a_text}\n   ⏱️ {q_elapsed:.1f}s")

    # ── Auto-generación ────────────────────────────────────────────────────────
    if n_auto > 0:
        auto_start = time.time()
        progress(len(questions) / max(total, 1), desc=f"Auto-generando {n_auto} Q&A…")
        auto_pairs = _auto_generate_qa(chunks, model, n_auto)
        auto_elapsed = time.time() - auto_start
        
        for i, (q, a, ev) in enumerate(auto_pairs):
            entries.append({
                "id":         f"{article_id}_auto{i:03d}_{uuid.uuid4().hex[:4]}",
                "article_id": article_id,
                "question":   q,
                "answer":     a,
                "evidence":   ev,
                "rag": {
                    "retriever_top_k": TOP_K_CHUNKS,
                    "model_teacher":   model,
                    "temperature":     0.3,
                    "prompt_version":  "v1_auto",
                },
            })
            q_text = q[:70] + "…" if len(q) > 70 else q
            a_text = a[:100] + "…" if len(a) > 100 else a
            log_lines.append(f"🤖 **[auto] Q:** {q_text}\n   **A:** {a_text}")
        
        log_lines.append(f"_Auto-generados: {len(auto_pairs)}/{n_auto} · ⏱️ {auto_elapsed:.1f}s_")

    progress(1.0, desc="Guardando…")
    save_start = time.time()
    _append_entries(entries)
    save_elapsed = time.time() - save_start

    total_elapsed = time.time() - start_time
    
    # Insertar sumario al principio
    log_lines.insert(2, f"**✅ Añadidos:** {len(entries)} ejemplos nuevos · ⏱️ **Total: {total_elapsed:.1f}s** (guardar: {save_elapsed:.1f}s)\n")
    
    return "\n\n".join(log_lines), _dataset_stats()


# ── Interfaz ──────────────────────────────────────────────────────────────────

def _load_models():
    try:
        return [m.model for m in ollama.list().models]
    except Exception:
        return []


def build_interface():
    with gr.Blocks(title="Dataset Builder") as demo:
        gr.Markdown("""# 🗂️ Dataset Builder para fine-tuning

Sube un documento y (opcionalmente) un archivo de preguntas. El RAG procesará
cada pregunta por separado y guardará el resultado en **`dataset.jsonl`**.

Puedes ejecutarlo **varias veces** con distintos artículos/preguntas para construir
el dataset de forma incremental. Cuando estés listo, ejecuta `finetune_qlora.py`.
""")

        with gr.Row():
            doc_in = gr.File(label="Documento (PDF o TXT)", type="filepath", file_count="single")
            q_in   = gr.File(label="Preguntas (TXT, una por línea) — opcional", type="filepath", file_count="single")

        with gr.Row():
            model_dd    = gr.Dropdown(choices=_load_models(), label="Modelo Ollama", interactive=True, scale=3)
            refresh_btn = gr.Button("↻ Refrescar modelos", scale=1)

        n_auto = gr.Slider(0, 20, value=5, step=1,
                           label="Q&A adicionales auto-generados por el modelo")

        with gr.Row():
            run_btn   = gr.Button("▶ Procesar y añadir al dataset", variant="primary")
            clear_btn = gr.Button("🗑️ Borrar dataset completo", variant="stop")

        stats_md = gr.Markdown(value=_dataset_stats())
        output   = gr.Textbox(label="Log", lines=22, interactive=False)

        refresh_btn.click(fn=lambda: gr.Dropdown(choices=_load_models()), outputs=model_dd)
        run_btn.click(
            fn=process_document,
            inputs=[doc_in, q_in, model_dd, n_auto],
            outputs=[output, stats_md],
            show_progress="minimal",
        )
        clear_btn.click(fn=_clear_dataset, outputs=output).then(fn=_dataset_stats, outputs=stats_md)

    return demo


if __name__ == "__main__":
    build_interface().launch()