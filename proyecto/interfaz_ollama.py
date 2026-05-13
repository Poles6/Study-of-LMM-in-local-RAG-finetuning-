from typing import List, Optional
import time
import PyPDF2
import ollama
import gradio as gr


MAX_DOC_CHARS = 12000


def load_models():
    try:
        result = ollama.list()
        return [m.model for m in result.models]
    except Exception as e:
        print("Error loading models:", e)
        return []


def _read_document(filepath: str) -> str:
    if not filepath:
        return ""

    filepath = str(filepath)
    lower_path = filepath.lower()

    if lower_path.endswith(".pdf"):
        try:
            with open(filepath, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                text = ""
                for page in reader.pages:
                    text += (page.extract_text() or "") + "\n"
                return text.strip()
        except Exception as e:
            return f"Error leyendo PDF: {e}"

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read().strip()
    except UnicodeDecodeError:
        try:
            with open(filepath, "r", encoding="latin-1") as f:
                return f.read().strip()
        except Exception as e:
            return f"Error leyendo archivo de texto: {e}"
    except Exception as e:
        return f"Error leyendo archivo: {e}"


def _read_questions(filepath: str) -> List[str]:
    if not filepath:
        return []

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
    except UnicodeDecodeError:
        try:
            with open(filepath, "r", encoding="latin-1") as f:
                text = f.read()
        except Exception:
            return []
    except Exception:
        return []

    return [q.strip() for q in text.splitlines() if q.strip()]


def _shrink_document(text: str, max_chars: int = MAX_DOC_CHARS) -> str:
    """Recorta el documento para no mandar demasiado contexto."""
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n\n[Documento recortado por longitud]"


def build_prompt(document: str, questions: List[str]) -> str:
    questions_block = "\n".join(
        [f"{i+1}. {q}" for i, q in enumerate(questions)]
    )

    return f"""
Eres un experto en análisis de artículos científicos.

Tu tarea es responder las preguntas usando únicamente la información contenida en el artículo.
Si una respuesta no aparece claramente en el texto, escribe: "No se indica en el documento".

Instrucciones:
- Responde en español.
- Mantén cada respuesta breve y directa.
- Sigue exactamente este formato:

1. Pregunta: ...
   Respuesta: ...

2. Pregunta: ...
   Respuesta: ...

<ARTICULO>
{document}

<PREGUNTAS>
{questions_block}
""".strip()


def answer_questions(document_file, question_file, model_name: Optional[str] = None) -> str:
    if not document_file or not question_file or not model_name:
        return "Por favor, sube el documento, el archivo de preguntas y selecciona un modelo."

    document_text = _read_document(document_file)
    if not document_text.strip():
        return "No se pudo leer el documento o está vacío."

    questions = _read_questions(question_file)
    if not questions:
        return "El archivo de preguntas está vacío o no contiene preguntas válidas."

    # Recorta el documento para evitar prompts enormes
    document_text = _shrink_document(document_text, MAX_DOC_CHARS)

    prompt = build_prompt(document_text, questions)

    start_time = time.time()
    try:
        response = ollama.generate(
            model=model_name,
            prompt=prompt,
            options={
                "temperature": 0.1,
                "num_predict": 1200,
            },
        )

        if hasattr(response, "response"):
            result = response.response.strip()
        elif isinstance(response, dict):
            result = response.get("response", "").strip()
        else:
            result = str(response).strip()

        elapsed = time.time() - start_time
        return f"{result}\n\n---\n⏱️ Tiempo: {elapsed:.1f}s"

    except Exception as e:
        elapsed = time.time() - start_time
        return f"Error al generar respuesta: {e}\n\n⏱️ Tiempo hasta error: {elapsed:.1f}s"


def build_interface():
    with gr.Blocks() as demo:
        gr.Markdown(
            """# Interfaz de Preguntas y Respuestas con Ollama

Sube un **documento** (PDF o TXT), un archivo de **preguntas** (una por línea),
selecciona un modelo local de Ollama y pulsa **Responder**.

Recomendación: usa `llama3:latest` o `mathstral:latest` para esta tarea.
"""
        )

        with gr.Row():
            doc_in = gr.File(
                label="Documento de contexto (PDF o TXT)",
                type="filepath",
                file_count="single"
            )
            q_in = gr.File(
                label="Archivo de preguntas (TXT)",
                type="filepath",
                file_count="single"
            )

        with gr.Row():
            model_dropdown = gr.Dropdown(
                choices=load_models(),
                label="Modelo disponible",
                interactive=True
            )
            refresh_btn = gr.Button("Refrescar modelos")

        run_btn = gr.Button("Responder")

        output = gr.Textbox(
            label="Respuestas",
            lines=25,
            interactive=False
        )

        refresh_btn.click(
            fn=lambda: gr.Dropdown(choices=load_models()),
            inputs=[],
            outputs=model_dropdown
        )

        run_btn.click(
            fn=answer_questions,
            inputs=[doc_in, q_in, model_dropdown],
            outputs=output,
            show_progress="minimal"
        )

    return demo


if __name__ == "__main__":
    app = build_interface()
    app.launch()