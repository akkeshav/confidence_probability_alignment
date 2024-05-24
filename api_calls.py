import config
from openai import OpenAI

client = OpenAI(
    api_key=config.api_key
)


def get_open_ai_response(engine_name, prompt, temp):
    return client.chat.completions.create(
        model=engine_name,
        messages=[
            {"role": "user", "content": prompt}
        ],
        logprobs=True,
        top_logprobs=5,
        temperature=temp
    )
