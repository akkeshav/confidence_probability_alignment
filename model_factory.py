from transformers import AutoTokenizer
from transformers import MistralForCausalLM

class ModelFactory():
    def __init__(self) -> None:
        self.supported_models = set(["HuggingFaceH4/zephyr-7b-beta"])

    def load(self, model_name: str):
        if model_name not in self.supported_models:
            raise NotImplemented(f"This model is currently not supported: {model_name}. Please select one of the following: {self.supported_models}")
        
        if model_name == "HuggingFaceH4/zephyr-7b-beta":
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = MistralForCausalLM.from_pretrained(model_name)
            return tokenizer, model
