import config
from openai import OpenAI
import variable_util
from model_factory import ModelFactory


def get_open_ai_response(engine_name, prompt, temp):
    client = OpenAI(api_key=config.api_key)
    return client.chat.completions.create(
        model=engine_name,
        messages=[
            {"role": "user", "content": prompt}
        ],
        logprobs=True,
        top_logprobs=5,
        temperature=temp
    )


def get_model_and_tokenizer(engine_name):
    model_factory = ModelFactory()

    return model_factory.load(variable_util.open_source_dict[engine_name])
