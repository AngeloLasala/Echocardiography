"""
Read the prediction and the real value of RWT and RST for the statistical analysis
"""
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

def read_single_trial(trial_dir):
    """
    read the single trial folder
    """
    print(os.listdir(trial_dir))

def main(args):
    """
    main of statistical analysis of RWT and RST
    """
    for trial in args.trials:
        trial_dir = os.path.join(args.model_path, trial)
        read_single_trial(trial_dir)
        

        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Read the prediction and the real value of RWT and RST for the statistical analysis')
    parser.add_argument('--model_path', type=str, default="/media/angelo/OS/Users/lasal/OneDrive - Scuola Superiore Sant'Anna/PhD_notes/Echocardiografy/trained_model/regression/Batch2/data_augumentation", help='Directory of models, i.e. trained_model')
    parser.add_argument('--trials', nargs='+', type=str, default=['trial_1'], help='trial number to analyse')

    args = parser.parse_args()

    main(args)