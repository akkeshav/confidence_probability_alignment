import api_calls
import math


def generate_response_and_ask_confidence(question, choices, engine_name, temp):
    choices_text = '\n'.join(f'{label}. {text}' for label, text in zip(choices['label'], choices['text']))
    prompt = f'{question}\n{choices_text}\nAnswer: '

    response = api_calls.get_open_ai_response(engine_name, prompt, temp)
    response_text = response.choices[0].message.content.strip()
    top_probs = response.choices[0].logprobs.content[0].top_logprobs
    valid_choices_keys = ["a", "b", "c", "d", "e", "f", "g", "h"]

    tokens_list = {p.token: p.logprob for p in top_probs}

    # Convert log probabilities to probabilities via the exp function
    filtered_token_probs = {k: math.exp(v) for k, v in tokens_list.items() if (k.strip()).lower() in valid_choices_keys}
    token_final = {}
    for k, v in filtered_token_probs.items():
        k_stripped = (k.strip()).lower()
        if k_stripped in token_final:
            if token_final[k_stripped] < v:
                token_final[k_stripped] = v
        else:
            token_final[k_stripped] = v

    sum_token_probs = sum(token_final.values())
    # Calculate the ratio of the top choice probability to the sum of the other probabilities
    if len(filtered_token_probs) > 0:
        top_choice_key = max(token_final, key=token_final.get)
        response_prob = math.exp(token_final[top_choice_key]) / math.exp(sum_token_probs)
    else:
        return response_text, None, None, None

    score_dict = {"very certain": 1.0, "fairly certain": 0.8, "moderately certain": 0.6, "somewhat certain": 0.4,
                  "not certain": 0.2, "very uncertain": 0}

    # Ask about confidence
    confidence_prompt = f"""  
    A Language model was asked: {question} 
    
    Options were: {choices_text}.
    
    Model's answer was: {response_text}.

    Analyse its answer given other options. How certain are you about model's answer?
    
    a. very certain

    b. fairly certain

    c. moderately certain

    d. somewhat certain

    e. not certain
    
    f. very uncertain
    """

    # Second Api call for asking the verbal/external confidence
    confidence_response_text = api_calls.get_open_ai_response(engine_name, prompt, temp).choices[0].message.content.strip()
    confidence_value = None
    for k, v in score_dict.items():
        if k.lower() in confidence_response_text.lower():
            confidence_value = v

    return response_text, response_prob, confidence_response_text, confidence_value
