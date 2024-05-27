# confidence_probability_alignment

![](./Intro_Diagram.png)
Illustration of **GPT-4's** responses to various questions, accompanied by their internal confidences and 
expressed certainty levels. Questions sourced from CommonsenseQA dataset.

## Setup

- Go to the project directory and use the following command
```commandline
pip install -r requirements.txt
```
- This command will install required libraries.

## Definitions
- we have two types of major model category - Open source and closed source models.
- Open source models - ["phi", "zephyr"]
- Closed source models - ['gpt4', 'text-davinci-003', 'text-davinci-002', 'text-davinci-001']
- datasets - ['commonsense_qa', 'openbookqa', 'qasc', 'riddle_sense', 'ai2_arc']

## Generate Internal Confidence and Verbalized Certainty
- Please use the following command on the terminal to get internal confidence and verbalized certainty
```commandline
python generate_alignment.py --model model_name --dataset dataset
```

- After using this command a new data directory will be created and it will
have the following directory structure.
my_project/
```
├── README.md
├── data/
│   ├── dataset_1 
│       ├── model_1.xlsx
│       ├── model_2.xlsx
│   ├── dataset_2 
│       ├──model_1.xlsx
│       ├──model_2.xlsx
│   .
│   .
│   .
├── utils.py
├── requirements.txt
├── variable_utils.txt
└── config.py
```

These model_1.xlsx, model_2.xlsx etc, contains both internal confidence 
and verbalized certainty along with actual textual responses.
