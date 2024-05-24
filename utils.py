import pandas as pd
from pathlib import Path
import os


def save_data(dataset, model, questions, responses, response_probs, confidence_text, confidence_values, answers):
    data = {'questions': questions, 'responses': responses, 'response_probs': response_probs,
            'confidence_text': confidence_text, 'confidence_values': confidence_values, 'answers': answers}
    df = pd.DataFrame(data)
    directory = f"data/{dataset}/"
    filename = f"{model}.xlsx"
    Path("data").mkdir(parents=True, exist_ok=True)
    Path(directory).mkdir(parents=True, exist_ok=True)
    file_path = os.path.join(directory, filename)
    try:
        df.to_excel(file_path, index=True)
    except Exception as e:
        print("Error encountered while creating file:", e)
