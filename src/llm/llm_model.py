from __future__ import annotations

import os
import threading
from typing import Any, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


MODEL_NAME = os.getenv("LLM_MODEL_NAME", "Qwen/Qwen2.5-1.5B-Instruct")
QUANTIZATION = os.getenv("LLM_QUANTIZATION", "4bit").strip().lower()

MAX_NEW_TOKENS = int(os.getenv("LLM_MAX_NEW_TOKENS", "640"))
MAX_INPUT_TOKENS = int(os.getenv("LLM_MAX_INPUT_TOKENS", "3072"))

DO_SAMPLE = os.getenv("LLM_DO_SAMPLE", "false").strip().lower() == "true"
TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))
TOP_P = float(os.getenv("LLM_TOP_P", "0.9"))
REPETITION_PENALTY = float(os.getenv("LLM_REPETITION_PENALTY", "1.08"))

HF_CACHE_DIR = os.getenv("HF_HOME", None)

_MODEL = None
_TOKENIZER = None
_LLM_WRAPPER = None
_LOCK = threading.RLock()


def _has_cuda() -> bool:
    return torch.cuda.is_available()


def _preferred_torch_dtype():
    if _has_cuda():
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32


def _build_quantization_config() -> Optional[BitsAndBytesConfig]:
    quant = QUANTIZATION

    if quant in {"", "none", "fp16", "fp32"}:
        return None

    if not _has_cuda():
        print(
            f"[WARN] LLM_QUANTIZATION={QUANTIZATION} nhưng torch không có CUDA. "
            f"Tự fallback về non-quantized load."
        )
        return None

    if quant == "8bit":
        return BitsAndBytesConfig(load_in_8bit=True)

    if quant == "4bit":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=_preferred_torch_dtype(),
        )

    raise ValueError(
        f"LLM_QUANTIZATION không hợp lệ: {QUANTIZATION}. "
        f"Dùng một trong: 4bit, 8bit, none"
    )


def _load_model_and_tokenizer():
    global _MODEL, _TOKENIZER

    if _MODEL is not None and _TOKENIZER is not None:
        return _MODEL, _TOKENIZER

    with _LOCK:
        if _MODEL is not None and _TOKENIZER is not None:
            return _MODEL, _TOKENIZER

        print("=" * 80)
        print(f"[INFO] Loading local LLM: {MODEL_NAME}")
        print(f"[INFO] Quantization mode: {QUANTIZATION}")
        print(f"[INFO] CUDA available: {_has_cuda()}")
        print(f"[INFO] Preferred dtype: {_preferred_torch_dtype()}")
        print(f"[INFO] Max input tokens: {MAX_INPUT_TOKENS}")
        print(f"[INFO] Max new tokens  : {MAX_NEW_TOKENS}")
        print(f"[INFO] Do sample       : {DO_SAMPLE}")
        print("=" * 80)

        quant_config = _build_quantization_config()

        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            cache_dir=HF_CACHE_DIR,
            trust_remote_code=False,
            use_fast=True,
        )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        load_kwargs = {
            "pretrained_model_name_or_path": MODEL_NAME,
            "cache_dir": HF_CACHE_DIR,
            "trust_remote_code": False,
            "dtype": _preferred_torch_dtype(),
        }

        if _has_cuda() or quant_config is not None:
            load_kwargs["device_map"] = "auto"

        if quant_config is not None:
            load_kwargs["quantization_config"] = quant_config

        model = AutoModelForCausalLM.from_pretrained(**load_kwargs)
        model.eval()

        # Đồng bộ generation_config để tránh warning do checkpoint mang sẵn
        # sampling params nhưng runtime hiện tại lại đang dùng greedy decoding.
        if hasattr(model, "generation_config") and model.generation_config is not None:
            model.generation_config.do_sample = DO_SAMPLE

            if DO_SAMPLE:
                model.generation_config.temperature = TEMPERATURE
                model.generation_config.top_p = TOP_P
            else:
                for attr in ("temperature", "top_p", "typical_p", "penalty_alpha"):
                    if hasattr(model.generation_config, attr):
                        try:
                            setattr(model.generation_config, attr, None)
                        except Exception:
                            pass

                if hasattr(model.generation_config, "top_k"):
                    try:
                        model.generation_config.top_k = None
                    except Exception:
                        pass

        try:
            mem = model.get_memory_footprint()
            print(f"[INFO] Model memory footprint: {mem / (1024 ** 3):.2f} GB")
        except Exception:
            pass

        _MODEL = model
        _TOKENIZER = tokenizer

        print("[DONE] Local LLM loaded successfully.")
        return _MODEL, _TOKENIZER


def _message_to_text(message: Any) -> str:
    if message is None:
        return ""

    if isinstance(message, str):
        return message

    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if text:
                    parts.append(str(text))
            else:
                parts.append(str(item))
        return "\n".join(x for x in parts if x).strip()

    if isinstance(message, dict):
        for key in ("content", "text", "prompt", "message"):
            if key in message and message[key] is not None:
                return str(message[key])

    return str(message)


def _coerce_prompt_to_text(prompt: Any) -> str:
    if prompt is None:
        return ""

    if isinstance(prompt, str):
        return prompt

    if hasattr(prompt, "to_string"):
        try:
            text = prompt.to_string()
            if isinstance(text, str) and text.strip():
                return text
        except Exception:
            pass

    if hasattr(prompt, "to_messages"):
        try:
            messages = prompt.to_messages()
            if isinstance(messages, list):
                text = "\n\n".join(
                    _message_to_text(m) for m in messages if _message_to_text(m).strip()
                ).strip()
                if text:
                    return text
        except Exception:
            pass

    if hasattr(prompt, "messages"):
        try:
            messages = getattr(prompt, "messages")
            if isinstance(messages, list):
                text = "\n\n".join(
                    _message_to_text(m) for m in messages if _message_to_text(m).strip()
                ).strip()
                if text:
                    return text
        except Exception:
            pass

    if isinstance(prompt, list):
        text = "\n\n".join(
            _message_to_text(x) for x in prompt if _message_to_text(x).strip()
        ).strip()
        if text:
            return text

    if isinstance(prompt, dict):
        for key in ("text", "prompt", "content", "input"):
            if key in prompt and prompt[key] is not None:
                return str(prompt[key])

    return str(prompt)


def _postprocess_output(text: str) -> str:
    if not text:
        return ""

    text = text.strip()

    prefixes = [
        "Trả lời:",
        "Câu trả lời:",
        "Answer:",
        "Final answer:",
    ]
    for prefix in prefixes:
        if text.startswith(prefix):
            text = text[len(prefix) :].strip()

    text = text.replace("[trong CONTEXT]", "trong tài liệu tham khảo")
    text = text.replace("trong CONTEXT", "trong tài liệu tham khảo")

    return text.strip()


class LocalHFLLM:
    def __init__(self):
        self.model, self.tokenizer = _load_model_and_tokenizer()

    def _build_input_text(self, prompt: str) -> str:
        try:
            messages = [
                {
                    "role": "user",
                    "content": prompt,
                }
            ]
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            return prompt

    def invoke(self, prompt: Any) -> str:
        prompt_text = _coerce_prompt_to_text(prompt)

        if not isinstance(prompt_text, str):
            prompt_text = str(prompt_text)

        prompt_text = prompt_text.strip()

        if not prompt_text:
            return "Prompt đầu vào đang rỗng nên tôi chưa thể sinh câu trả lời."

        input_text = self._build_input_text(prompt_text)

        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_INPUT_TOKENS,
        )

        model_device = next(self.model.parameters()).device
        inputs = {k: v.to(model_device) for k, v in inputs.items()}

        generate_kwargs = {
            **inputs,
            "max_new_tokens": MAX_NEW_TOKENS,
            "do_sample": DO_SAMPLE,
            "repetition_penalty": REPETITION_PENALTY,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "generation_config": self.model.generation_config,
        }

        if DO_SAMPLE:
            generate_kwargs["temperature"] = TEMPERATURE
            generate_kwargs["top_p"] = TOP_P

        with torch.inference_mode():
            output_ids = self.model.generate(**generate_kwargs)

        generated_ids = output_ids[0][inputs["input_ids"].shape[1] :]
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        return _postprocess_output(text)

    def __call__(self, prompt: Any) -> str:
        return self.invoke(prompt)


def load_llm() -> LocalHFLLM:
    global _LLM_WRAPPER

    if _LLM_WRAPPER is not None:
        return _LLM_WRAPPER

    with _LOCK:
        if _LLM_WRAPPER is not None:
            return _LLM_WRAPPER

        _LLM_WRAPPER = LocalHFLLM()
        return _LLM_WRAPPER


def get_llm() -> LocalHFLLM:
    return load_llm()


def build_llm() -> LocalHFLLM:
    return load_llm()


def create_llm() -> LocalHFLLM:
    return load_llm()


def generate_answer(prompt: Any) -> str:
    return load_llm().invoke(prompt)


def generate(prompt: Any) -> str:
    return load_llm().invoke(prompt)


def run_llm(prompt: Any) -> str:
    return load_llm().invoke(prompt)


if __name__ == "__main__":
    sample_prompt = (
        "Bạn là trợ lý hỗ trợ nội bộ.\n\n"
        "Câu hỏi: Làm thế nào để đăng nhập bằng passkey?\n\n"
        "CONTEXT:\n"
        "[1] signing in with a passkey\n"
        "CONTENT:\n"
        "1. Navigate to the login page.\n"
        "2. Click Sign in with a passkey.\n"
        "3. Follow the prompts.\n"
    )
    llm = load_llm()
    print(llm.invoke(sample_prompt))