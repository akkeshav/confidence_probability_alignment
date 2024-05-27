import argparse
import utils
import variable_util

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Confidence_probability_alignment")
    parser.add_argument('--model', type=str, required=True, choices=['gpt4', 'text-davinci-03',
                                                                     'text-davinci-02', 'text-davinci-01', 'phi',
                                                                     'zephyr'],
                        help='choose a model')
    parser.add_argument('--dataset', type=str, required=True, choices=['commonsense_qa', 'openbookqa',
                                                                       'qasc', 'riddle_sense', 'ai2_arc'],
                        help='dataset names')

    model = parser.parse_args().model
    dataset = parser.parse_args().dataset


