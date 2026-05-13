"""
chat_finetuned.py — Chat simple para testear el modelo fine-tuned.

Usa Ollama igual que interfaz_ollama.py pero sin subir documentos:
el modelo debe responder por sí solo, demostrando lo que aprendió.

Flujo típico de uso:
  1. Entrenar con finetune_qlora.py  →  adapter_out/
  2. Importar en Ollama:
       ollama pull <base_model>
       ollama create mi_modelo_ft -f adapter_out/Modelfile
  3. Abrir este chat y seleccionar "mi_modelo_ft"
  4. Hacer preguntas similares a las del dataset y comparar respuestas

También puedes usar este chat para comparar lado a lado el modelo
base y el fine-tuned seleccionando uno u otro en el desplegable.
"""

import time

import ollama
import gradio as gr

DEFAULT_SYSTEM = (
    "Eres un experto en análisis de artículos científicos. "
    "Responde siempre en español de forma breve y directa. "
    "Si no sabes la respuesta, di: No se indica en el documento."
)


def _load_models():
    try:
        return [m.model for m in ollama.list().models]
    except Exception:
        return []


def chat(message: str, history: list, model: str, system_prompt: str):
    if not message.strip():
        return "", history
    if not model:
        history.append((message, "⚠️ Selecciona un modelo primero."))
        return "", history

    start_time = time.time()
    
    # Construir historial completo
    messages = []
    if system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt})
    for user_msg, assistant_msg in history:
        messages.append({"role": "user",      "content": user_msg})
        messages.append({"role": "assistant", "content": assistant_msg})
    messages.append({"role": "user", "content": message})

    try:
        response = ollama.chat(
            model=model,
            messages=messages,
            options={"temperature": 0.2, "num_predict": 500},
        )
        reply = response.message.content.strip()
        elapsed = time.time() - start_time
        reply += f"\n\n_⏱️ {elapsed:.1f}s_"
    except Exception as e:
        elapsed = time.time() - start_time
        reply = f"❌ Error: {e}\n\n_⏱️ {elapsed:.1f}s_"

    history.append((message, reply))
    return "", history


def build_interface():
    with gr.Blocks(title="Chat — Modelo Fine-tuned") as demo:
        gr.Markdown("""# 💬 Chat para testear el modelo fine-tuned

Habla directamente con el modelo **sin adjuntar documentos**.
Selecciona tu modelo fine-tuned en el desplegable y haz preguntas
similares a las del entrenamiento para evaluar lo que aprendió.

> **Tip:** compara el modelo base vs. el fine-tuned cambiando el desplegable.
""")

        with gr.Row():
            model_dd    = gr.Dropdown(choices=_load_models(), label="Modelo Ollama",
                                      interactive=True, scale=4)
            refresh_btn = gr.Button("↻", scale=1, min_width=60)

        system_box = gr.Textbox(
            value=DEFAULT_SYSTEM,
            label="System prompt (editable)",
            lines=2,
        )

        chatbot = gr.Chatbot(label="Conversación", height=430, bubble_full_width=False)

        with gr.Row():
            msg_box    = gr.Textbox(label="Pregunta", placeholder="Escribe tu pregunta…",
                                    lines=2, scale=5)
            submit_btn = gr.Button("Enviar", variant="primary", scale=1, min_width=80)

        clear_btn = gr.Button("🗑️ Limpiar conversación")

        # ── Bindings ──────────────────────────────────────────────────────────
        refresh_btn.click(fn=lambda: gr.Dropdown(choices=_load_models()), outputs=model_dd)

        submit_btn.click(
            fn=chat,
            inputs=[msg_box, chatbot, model_dd, system_box],
            outputs=[msg_box, chatbot],
            show_progress="minimal",
        )
        msg_box.submit(
            fn=chat,
            inputs=[msg_box, chatbot, model_dd, system_box],
            outputs=[msg_box, chatbot],
            show_progress="minimal",
        )
        clear_btn.click(fn=lambda: ([], ""), outputs=[chatbot, msg_box])

    return demo


if __name__ == "__main__":
    build_interface().launch()