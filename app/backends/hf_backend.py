from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList
import math
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
        stop: list[str] = None,
        **kwargs
    ):
        include_token_entropy = bool(kwargs.get("include_token_entropy", False))
        if not self.model:
            return "CI MOCK TEXT", {
                "output_tokens": 0,
                "token_entropy": [],
                "token_trace": [],
                "token_entropy_available": False,
            }

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        input_tokens = inputs["input_ids"].shape[1]
        repetition_penalty = float(kwargs.get("repetition_penalty", 1.1))
        no_repeat_ngram_size = int(kwargs.get("no_repeat_ngram_size", 3))

        class StopOnStrings(StoppingCriteria):
            def __init__(self, stop_strings, tokenizer):
                self.stop_strings = stop_strings
                self.tokenizer = tokenizer

            def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
                decoded = self.tokenizer.decode(input_ids[0][-10:], skip_special_tokens=True)
                return any(s in decoded for s in self.stop_strings)

        stopping_criteria = StoppingCriteriaList([StopOnStrings(stop, self.tokenizer)]) if stop else None

        generate_kwargs = {
            **inputs,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "do_sample": do_sample,
            "repetition_penalty": repetition_penalty,
            "no_repeat_ngram_size": no_repeat_ngram_size,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.eos_token_id,
            "stopping_criteria": stopping_criteria,
        }
        if include_token_entropy:
            generate_kwargs["return_dict_in_generate"] = True
            generate_kwargs["output_scores"] = True

        with torch.no_grad():
            outputs = self.model.generate(**generate_kwargs)

        if include_token_entropy:
            sequence = outputs.sequences[0]
            generated_ids = sequence[input_tokens:]
        else:
            generated_ids = outputs[0][input_tokens:]
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        meta = {"output_tokens": len(generated_ids), "input_tokens": input_tokens}
        if include_token_entropy:
            token_trace = self._build_token_trace(generated_ids, outputs.scores)
            meta["token_trace"] = token_trace
            meta["token_entropy"] = [
                {
                    "text": token["text"],
                    "entropy": token["entropy"],
                }
                for token in token_trace
            ]
            meta["token_entropy_available"] = True

        return text, meta

    def _build_token_trace(self, generated_ids: torch.Tensor, scores: tuple[torch.Tensor, ...]):
        token_trace = []
        vocab_size = self.model.config.vocab_size if self.model is not None else 0
        log_vocab = math.log(vocab_size) if vocab_size and vocab_size > 1 else None

        for token_id, score in zip(generated_ids.tolist(), scores):
            probs = torch.softmax(score[0], dim=-1)
            log_probs = torch.log_softmax(score[0], dim=-1)
            raw_entropy = float(-(probs * torch.log(probs.clamp_min(1e-12))).sum().item())
            normalized_entropy = raw_entropy / log_vocab if log_vocab else 0.0
            token_text = self.tokenizer.decode(
                [token_id],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            token_trace.append(
                {
                    "text": token_text,
                    "entropy": round(normalized_entropy, 6),
                    "token_id": int(token_id),
                    "logprob": round(float(log_probs[token_id].item()), 6),
                }
            )

        return token_trace
