import api_calls
import math
from datasets import load_dataset
import utils
import time
import tqdm
import variable_util
import torch.nn.functional as F
import torch


def generate_verbalized_certainty(question, choices_text, response_text, engine_name, temp):
    score_dict = {"very certain": 1.0, "fairly certain": 0.8, "moderately certain": 0.6, "somewhat certain": 0.4,
                  "not certain": 0.2, "very uncertain": 0}

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
    confidence_response_text = api_calls.get_open_ai_response(engine_name, confidence_prompt,
                                                              temp).choices[0].message.content.strip()
    confidence_value = None
    for k, v in score_dict.items():
        if k.lower() in confidence_response_text.lower():
            confidence_value = v

    return confidence_response_text, confidence_value


def closed_source_models(prompt, engine_name, temp):
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
        response_prob = None

    return response_text, response_prob


def open_source_models(prompt, engine_name, choices_list):
    tokenizer, model = api_calls.get_model_and_tokenizer(engine_name)

    choices_ids = []
    for i in range(len(choices_list)):
        ids = tokenizer.encode(choices_list[i])
        choices_ids.append((choices_list[i], ids))

    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids
    detailed_output = model(prompt_ids)
    probabilities = F.softmax(detailed_output.logits[0], dim=-1)

    probabilities = probabilities[-1]

    current_highest_prob_label = None
    current_highest_prob = None

    for choice_id in choices_ids:
        label, ids = choice_id
        label_probs = torch.gather(probabilities, dim=0, index=torch.tensor(ids))
        label_probs = torch.sum(label_probs).item()

        if (current_highest_prob_label is None) and (current_highest_prob is None):
            current_highest_prob_label = label
            current_highest_prob = label_probs

        elif current_highest_prob < label_probs:
            current_highest_prob = label_probs
            current_highest_prob_label = label

    return current_highest_prob_label, current_highest_prob


def generate_internal_confidence(prompt, choices, engine_name, temp):
    return closed_source_models(prompt, engine_name, temp) \
        if engine_name not in variable_util.open_source_models else \
        open_source_models(prompt, engine_name, choices['label'])


def generate_response_and_ask_confidence(question, choices, engine_name, temp):
    choices_text = '\n'.join(f'{label}. {text}' for label, text in zip(choices['label'], choices['text']))
    prompt = f'{question}\n{choices_text}\nAnswer: '

    response_text, response_prob = generate_internal_confidence(prompt, choices, engine_name, temp)
    confidence_response_text, confidence_value = \
        generate_verbalized_certainty(question, choices_text, response_text, engine_name, temp)

    return response_text, response_prob, confidence_response_text, confidence_value


def get_raw_data(dataset, model):
    dataset = load_dataset(dataset) if dataset != "ai2_arc" else load_dataset(dataset, 'ARC-Challenge')
    dataset = dataset.shuffle(seed=80)
    subset_size = 926 if dataset == 'qasc' else 1000
    temperature = 0.2

    # Initialize lists to store the results
    questions = []
    responses = []
    confidence_values = []
    confidence_text = []
    response_probs = []
    answers = []

    # Iterate over the subset of the dataset
    # Assuming you have loaded the dataset and 'subset_size' is the size of your desired subset

    split = 'train' if dataset == 'openbookqa' or dataset == 'ai2_arc' else 'validation'

    for i in tqdm(range(subset_size)):
        time.sleep(4)
        example = {key: value[0] for key, value in dataset[split][i:i + 1].items()}
        response_text, response_prob, confidence_response_text, confidence_value = generate_response_and_ask_confidence(
            example['question'], example['choices'], model, temperature)
        # Append the results to the lists
        questions.append(example['question'])
        responses.append(response_text)
        confidence_text.append(confidence_response_text)
        response_probs.append(response_prob)
        confidence_values.append(confidence_value)
        answers.append(example['answerKey'])

    utils.save_data(dataset, model, questions, responses, response_probs, confidence_text, confidence_values, answers)
