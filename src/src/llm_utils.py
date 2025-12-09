from dataclasses import dataclass
from typing import Optional
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM

@dataclass
class LLMConfig:
    model_name: str = "google/flan-t5-small"
    max_new_tokens: int = 64
    temperature: float = 0.2
    device: Optional[str] = None

class LLM:
    def __init__(self, config: LLMConfig):
        self.config = config
        self.device = config.device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Try Seq2Seq first (good for instruction-style prompts)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(config.model_name).to(self.device)
            self.is_seq2seq = True
        except Exception:
            self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(config.model_name).to(self.device)
            self.is_seq2seq = False

    def generate(self, prompt: str) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True).to(self.device)

        if self.is_seq2seq:
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=True,
                temperature=self.config.temperature,
            )
            text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            return text.strip()

        # Causal LM fallback
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.config.max_new_tokens,
            do_sample=True,
            temperature=self.config.temperature,
            pad_token_id=self.tokenizer.eos_token_id
        )
        text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

        # Remove the prompt prefix if it appears verbatim
        if text.startswith(prompt):
            text = text[len(prompt):]
        return text.strip()
