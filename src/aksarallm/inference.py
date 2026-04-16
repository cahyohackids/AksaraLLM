"""Inference helpers and CLI for AksaraLLM chat models."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

DEFAULT_MODEL_NAME = "AksaraLLM/aksarallm-1.5b-v2"
DEFAULT_SYSTEM_PROMPT = (
    "Kamu adalah AksaraLLM, asisten AI berbahasa Indonesia yang cerdas, sopan, "
    "dan membantu. Jawab dengan jelas, jujur, dan ringkas bila memungkinkan."
)

__all__ = [
    "AksaraChatSession",
    "AksaraLLM",
    "GenerationSettings",
    "DEFAULT_MODEL_NAME",
    "DEFAULT_SYSTEM_PROMPT",
    "main",
]


def _load_runtime():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise RuntimeError(
            "PyTorch belum terpasang. Install runtime inference dengan "
            "`pip install -e \".[runtime]\"` atau `pip install -r requirements.txt`."
        ) from exc

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise RuntimeError(
            "Transformers belum terpasang. Install runtime inference dengan "
            "`pip install -e \".[runtime]\"` atau `pip install -r requirements.txt`."
        ) from exc

    return torch, AutoTokenizer, AutoModelForCausalLM


def _default_device(torch_module: Any) -> str:
    if torch_module.cuda.is_available():
        return "cuda"
    if getattr(torch_module.backends, "mps", None) and torch_module.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass(frozen=True)
class GenerationSettings:
    """Runtime generation settings for chat inference."""

    temperature: float = 0.7
    top_p: float = 0.9
    max_new_tokens: int = 512
    repetition_penalty: float = 1.15


class AksaraChatSession:
    """Convenience wrapper around a Hugging Face chat model."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        *,
        device: str = "auto",
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        max_history_turns: int = 6,
        max_input_tokens: int = 4096,
        trust_remote_code: bool = True,
    ) -> None:
        torch_module, tokenizer_cls, model_cls = _load_runtime()
        self._torch = torch_module
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.max_history_turns = max_history_turns
        self.max_input_tokens = max_input_tokens
        self.trust_remote_code = trust_remote_code
        self.history: list[tuple[str, str]] = []

        resolved_device = _default_device(torch_module) if device == "auto" else device
        self.device = resolved_device
        torch_dtype = (
            torch_module.float16
            if resolved_device in {"cuda", "mps"}
            else torch_module.float32
        )

        load_kwargs: dict[str, Any] = {
            "trust_remote_code": trust_remote_code,
            "torch_dtype": torch_dtype,
        }
        if device == "auto" and resolved_device == "cuda":
            load_kwargs["device_map"] = "auto"

        self.tokenizer = tokenizer_cls.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
        )
        self.model = model_cls.from_pretrained(model_name, **load_kwargs)
        if "device_map" not in load_kwargs:
            self.model.to(resolved_device)
        self.model.eval()
        self._input_device = next(self.model.parameters()).device

    def _iter_history_messages(self, history: Sequence[Any] | None) -> Iterable[dict[str, str]]:
        turns = history if history is not None else self.history
        if not turns:
            return []

        trimmed = list(turns)[-self.max_history_turns :]
        if isinstance(trimmed[0], dict):
            messages: list[dict[str, str]] = []
            for item in trimmed:
                role = item.get("role")
                content = item.get("content")
                if role in {"user", "assistant"} and isinstance(content, str) and content.strip():
                    messages.append({"role": role, "content": content})
            return messages

        messages = []
        for turn in trimmed:
            if not isinstance(turn, (list, tuple)) or len(turn) != 2:
                continue
            user_msg, assistant_msg = turn
            if isinstance(user_msg, str) and user_msg.strip():
                messages.append({"role": "user", "content": user_msg})
            if isinstance(assistant_msg, str) and assistant_msg.strip():
                messages.append({"role": "assistant", "content": assistant_msg})
        return messages

    def build_messages(
        self,
        message: str,
        *,
        history: Sequence[Any] | None = None,
        system_prompt: str | None = None,
    ) -> list[dict[str, str]]:
        text = message.strip()
        if not text:
            raise ValueError("Pesan tidak boleh kosong.")

        prompt = system_prompt or self.system_prompt
        messages = [{"role": "system", "content": prompt}]
        messages.extend(self._iter_history_messages(history))
        messages.append({"role": "user", "content": text})
        return messages

    def generate_reply(
        self,
        message: str,
        *,
        history: Sequence[Any] | None = None,
        system_prompt: str | None = None,
        settings: GenerationSettings | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        max_new_tokens: int | None = None,
        repetition_penalty: float | None = None,
    ) -> str:
        effective = settings or GenerationSettings()
        if temperature is not None:
            effective = GenerationSettings(
                temperature=temperature,
                top_p=effective.top_p if top_p is None else top_p,
                max_new_tokens=effective.max_new_tokens
                if max_new_tokens is None
                else max_new_tokens,
                repetition_penalty=effective.repetition_penalty
                if repetition_penalty is None
                else repetition_penalty,
            )
        elif any(
            value is not None
            for value in (top_p, max_new_tokens, repetition_penalty)
        ):
            effective = GenerationSettings(
                temperature=effective.temperature,
                top_p=effective.top_p if top_p is None else top_p,
                max_new_tokens=effective.max_new_tokens
                if max_new_tokens is None
                else max_new_tokens,
                repetition_penalty=effective.repetition_penalty
                if repetition_penalty is None
                else repetition_penalty,
            )

        messages = self.build_messages(
            message,
            history=history,
            system_prompt=system_prompt,
        )
        prompt_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        model_inputs = self.tokenizer(
            prompt_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_tokens,
        )
        model_inputs = {
            name: value.to(self._input_device)
            for name, value in model_inputs.items()
        }

        generate_kwargs: dict[str, Any] = {
            "max_new_tokens": int(effective.max_new_tokens),
            "repetition_penalty": float(effective.repetition_penalty),
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        if effective.temperature > 0:
            generate_kwargs.update(
                {
                    "do_sample": True,
                    "temperature": float(effective.temperature),
                    "top_p": float(effective.top_p),
                }
            )
        else:
            generate_kwargs["do_sample"] = False

        with self._torch.no_grad():
            outputs = self.model.generate(**model_inputs, **generate_kwargs)

        generated = outputs[0][model_inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()

    def chat(
        self,
        message: str,
        *,
        system_prompt: str | None = None,
        settings: GenerationSettings | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        max_new_tokens: int | None = None,
        repetition_penalty: float | None = None,
    ) -> str:
        reply = self.generate_reply(
            message,
            history=self.history,
            system_prompt=system_prompt,
            settings=settings,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty,
        )
        self.history.append((message, reply))
        self.history = self.history[-self.max_history_turns :]
        return reply

    def reset(self) -> None:
        """Clear in-memory conversation history."""
        self.history.clear()


# Backward-compatible alias for older imports: ``from aksarallm.inference import AksaraLLM``.
AksaraLLM = AksaraChatSession


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Interactive CLI untuk AksaraLLM.")
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
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperature sampling. Gunakan 0 untuk jawaban deterministik.",
    )
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
        help="Jumlah turn chat yang disimpan di memori.",
    )
    parser.add_argument(
        "--max-input-tokens",
        type=int,
        default=4096,
        help="Batas token prompt yang dikirim ke model.",
    )
    parser.add_argument(
        "--message",
        help="Mode sekali jalan. Jika diisi, CLI tidak masuk mode interaktif.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    session = AksaraChatSession(
        model_name=args.model,
        device=args.device,
        system_prompt=args.system_prompt,
        max_history_turns=args.max_history_turns,
        max_input_tokens=args.max_input_tokens,
    )
    settings = GenerationSettings(
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        repetition_penalty=args.repetition_penalty,
    )

    if args.message:
        print(session.chat(args.message, settings=settings))
        return 0

    print("AksaraLLM Chat")
    print(f"Model  : {args.model}")
    print(f"Device : {session.device}")
    print("Perintah: /reset, /history, /exit")
    print()

    while True:
        try:
            user_text = input("Anda> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0

        if not user_text:
            continue
        if user_text in {"/exit", "/quit"}:
            return 0
        if user_text == "/reset":
            session.reset()
            print("Riwayat chat dibersihkan.")
            continue
        if user_text == "/history":
            if not session.history:
                print("Riwayat masih kosong.")
                continue
            for index, (prompt, reply) in enumerate(session.history, start=1):
                print(f"{index}. Anda : {prompt}")
                print(f"   Aksara: {reply}")
            continue

        reply = session.chat(user_text, settings=settings)
        print(f"Aksara> {reply}\n")


if __name__ == "__main__":
    raise SystemExit(main())
