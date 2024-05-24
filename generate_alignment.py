import argparse
import utils
import variable_util

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Subtle bias")
    parser.add_argument('--model', type=str, required=True, choices=['gpt4', 'text-davinci-03',
                        'text-davinci-02', 'phi', 'zypher'], help='choose a model')
    parser.add_argument('--dataset', type=str, required=True, choices=['commonsense_qa', 'openbookqa',
                                                                       'qasc', 'riddle_sense', 'ai2_arc'],
                        help='dataset names')


    evaluatee_model = parser.parse_args().evaluatee_model
    evaluator_model = parser.parse_args().evaluator_model
    task = parser.parse_args().task
