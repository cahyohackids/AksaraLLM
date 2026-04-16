"""Gradio Web UI for AksaraLLM chat models."""

from __future__ import annotations

import argparse
import inspect
from typing import Sequence

from .inference import (
    DEFAULT_MODEL_NAME,
    DEFAULT_SYSTEM_PROMPT,
    AksaraChatSession,
)

EXAMPLES = [
    "Siapa kamu?",
    "Jelaskan Pancasila dengan bahasa yang mudah dipahami pelajar SMP.",
    "Buatkan outline artikel tentang UMKM digital di Indonesia.",
    "Tolong jelaskan perbedaan AI, machine learning, dan deep learning.",
]


def _load_gradio():
    try:
        import gradio as gr
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise RuntimeError(
            "Gradio belum terpasang. Jalankan `pip install -e \".[demo]\"` "
            "atau `pip install -r requirements.txt`."
        ) from exc
    return gr


def _supports_kwarg(callable_obj, name: str) -> bool:
    try:
        return name in inspect.signature(callable_obj).parameters
    except (TypeError, ValueError):
        return False


def _filter_supported_kwargs(callable_obj, kwargs: dict) -> dict:
    return {
        key: value
        for key, value in kwargs.items()
        if _supports_kwarg(callable_obj, key)
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Web UI untuk AksaraLLM.")
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME, help="Hugging Face model id.")
    parser.add_argument(
        "--device",
        default="auto",
        help="Device untuk inference: auto, cpu, cuda, atau mps.",
    )
    parser.add_argument(
        "--system-prompt",
        default=DEFAULT_SYSTEM_PROMPT,
        help="System prompt default.",
    )
    parser.add_argument("--host", default="0.0.0.0", help="Host Gradio.")
    parser.add_argument("--port", type=int, default=7860, help="Port Gradio.")
    parser.add_argument("--share", action="store_true", help="Aktifkan public share link.")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature sampling.")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p sampling.")
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Jumlah token jawaban maksimum.",
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.15,
        help="Penalty untuk mengurangi repetisi.",
    )
    parser.add_argument(
        "--max-history-turns",
        type=int,
        default=6,
        help="Jumlah turn chat yang dipakai saat membangun konteks.",
    )
    parser.add_argument(
        "--max-input-tokens",
        type=int,
        default=4096,
        help="Batas token prompt yang dikirim ke model.",
    )
    parser.add_argument(
        "--defer-model-load",
        action="store_true",
        help="Tunda load model sampai ada request pertama.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    gr = _load_gradio()
    session_holder: dict[str, AksaraChatSession | None] = {"session": None}

    def load_session() -> AksaraChatSession:
        session = session_holder["session"]
        if session is None:
            session = AksaraChatSession(
                model_name=args.model,
                device=args.device,
                system_prompt=args.system_prompt,
                max_history_turns=args.max_history_turns,
                max_input_tokens=args.max_input_tokens,
            )
            session_holder["session"] = session
        return session

    if not args.defer_model_load:
        load_session()

    def respond(
        message: str,
        history: list[dict[str, str]],
        system_prompt: str,
        temperature: float,
        top_p: float,
        max_new_tokens: int,
        repetition_penalty: float,
    ) -> str:
        session = load_session()
        return session.generate_reply(
            message,
            history=history,
            system_prompt=system_prompt,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=int(max_new_tokens),
            repetition_penalty=repetition_penalty,
        )

    css = """
    .gradio-container {
        max-width: 1080px !important;
        margin: 0 auto !important;
    }
    .app-shell {
        border: 1px solid rgba(148, 163, 184, 0.25);
        border-radius: 24px;
        background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
        box-shadow: 0 24px 80px rgba(15, 23, 42, 0.08);
        padding: 24px;
    }
    .app-hero {
        margin-bottom: 16px;
    }
    """

    with gr.Blocks(
        css=css,
        title="AksaraLLM Web UI",
        theme=gr.themes.Soft(primary_hue="rose", neutral_hue="slate"),
    ) as demo:
        chatbot_kwargs = _filter_supported_kwargs(
            gr.Chatbot.__init__,
            {
            "height": 540,
            "show_copy_button": True,
            "label": "AksaraLLM",
            "type": "messages",
            },
        )

        chat_interface_kwargs = _filter_supported_kwargs(
            gr.ChatInterface.__init__,
            {
            "fn": respond,
            "chatbot": gr.Chatbot(**chatbot_kwargs),
            "textbox": gr.Textbox(
                placeholder="Tulis pertanyaan Anda di sini...",
                container=False,
            ),
            "additional_inputs": [
                gr.Textbox(
                    value=args.system_prompt,
                    label="System Prompt",
                    lines=2,
                ),
                gr.Slider(0.0, 1.5, value=args.temperature, step=0.1, label="Temperature"),
                gr.Slider(0.1, 1.0, value=args.top_p, step=0.05, label="Top-P"),
                gr.Slider(64, 1024, value=args.max_new_tokens, step=64, label="Max New Tokens"),
                gr.Slider(
                    1.0,
                    2.0,
                    value=args.repetition_penalty,
                    step=0.05,
                    label="Repetition Penalty",
                ),
            ],
                "examples": [[example] for example in EXAMPLES],
                "fill_height": True,
                "type": "messages",
            },
        )

        with gr.Column(elem_classes=["app-shell"]):
            gr.Markdown(
                f"""
                <div class="app-hero">
                  <h1>🇮🇩 AksaraLLM Chat</h1>
                  <p>Chat cepat untuk model <code>{args.model}</code>.</p>
                  <p>CPU tetap bisa dipakai, tetapi pengalaman terbaik biasanya ada di GPU atau Apple Silicon yang memadai.</p>
                  <p>{'Model akan di-load saat request pertama.' if args.defer_model_load else 'Model di-load saat startup.'}</p>
                </div>
                """
            )

            gr.ChatInterface(**chat_interface_kwargs)

            gr.Markdown(
                """
                Tips:
                - Gunakan pertanyaan singkat dan spesifik untuk respons yang lebih konsisten.
                - Jika jawaban mulai melenceng, kosongkan chat lalu mulai topik baru.
                - Untuk deployment publik, siapkan evaluasi dan guardrail terpisah sebelum dipublikasikan.
                """
            )

    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        show_error=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
