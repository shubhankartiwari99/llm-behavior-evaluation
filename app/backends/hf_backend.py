from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os


class HFBackend:

    def __init__(self):
        # Allow CI environments to skip weight downloading on instantiate
        if os.environ.get("SKIP_HF_LOAD") == "1":
            self.model = None
            self.tokenizer = None
            return

        model_id = os.environ.get(
            "HF_MODEL_ID",
            "Qwen/Qwen2.5-7B-Instruct"
        )

        hf_token = os.environ.get("HF_TOKEN", None)

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            token=hf_token
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            token=hf_token
        )

        self.model.eval()

    def generate(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        do_sample: bool = False,
        **kwargs
    ):
        if not self.model:
            return "CI MOCK TEXT", 0

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        input_tokens = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                eos_token_id=self.tokenizer.eos_token_id
            )

        generated_ids = outputs[0][input_tokens:]
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        return text, len(generated_ids)
