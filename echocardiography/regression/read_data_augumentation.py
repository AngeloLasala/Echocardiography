"""
Read the prediction and the real value of RWT and RST for the statistical analysis
"""
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon, friedmanchisquare
import json

from scipy import stats


def read_single_trial(model_path, trial):
    """
    read the single trial folder
    """
    trial_dir = os.path.join(model_path, trial)
    ## read the args fiel
    if 'fold_1' in os.listdir(trial_dir):
        args_file = os.path.join(trial_dir, 'fold_1', 'args.json')
        with open(args_file) as f:
            args = json.load(f)

        p= args['percentace']

        mpe_list, mae_list = [], []
        rwt_error_list, rst_error_list = [], []
        for fold in os.listdir(trial_dir):
            fold_path = os.path.join(trial_dir, fold)

            ## load label and prediction
            heatmap_file = os.path.join(fold_path, 'heatmaps_ellipses_distances_label_cm_list.npy')
            prediction_file = os.path.join(fold_path, 'heatmaps_ellipses_distances_output_cm_list.npy')
            label = np.load(heatmap_file, allow_pickle=True)
            prediction = np.load(prediction_file, allow_pickle=True)

            mpe = np.abs(label - prediction) / label
            mae = np.abs(label - prediction)
            rwt_label, rwt_prediction = (2 * label[:, 0]) / label[:,1], (2 * prediction[:, 0]) / prediction[:, 1]
            rst_label, rst_prediction = (2 * label[:, 2]) / label[:,1], (2 * prediction[:, 2]) / prediction[:, 1]
            rwt_error = np.abs(rwt_label - rwt_prediction)
            rst_error = np.abs(rst_label - rst_prediction)

            mpe_list.append(mpe)
            mae_list.append(mae)
            rwt_error_list.append(rwt_error)
            rst_error_list.append(rst_error)


        mpe_list_c = np.concatenate(mpe_list)
        mae_list_c = np.concatenate(mae_list)
        rwt_error_list_c = np.concatenate(rwt_error_list)
        rst_error_list_c = np.concatenate(rst_error_list)


        single_exp_dict = {'mpe': mpe_list_c, 'mae': mae_list_c, 'rwt_error': rwt_error_list_c, 'rst_error': rst_error_list_c,
                           'mpe_fold':mpe_list, 'mae_fold':mae_list, 'rwt_error_fold':rwt_error_list, 'rst_error_fold':rst_error_list,
                           'percentace': p}

        return single_exp_dict

    else:
        ## value error if the fold_1 is not present
        raise ValueError('fold_1 is not present in the directory')


def friedman_and_post_hoc(args, experiments):
    """
    Conmpute the friedman test and the post hoc test
    """
    mpe = {trial: experiments[trial]['mpe'] for trial in args.trials}
    mae = {trial: experiments[trial]['mae'] for trial in args.trials}
    rwt_error = {trial: experiments[trial]['rwt_error'] for trial in args.trials}
    rst_error = {trial: experiments[trial]['rst_error'] for trial in args.trials}

    ## compute the Friedman test
    rwt_friedman = friedmanchisquare(*[rwt_error[trial] for trial in args.trials])
    rst_friedman = friedmanchisquare(*[rst_error[trial] for trial in args.trials])
    print(f'Friedman test for MPE: statistic {rwt_friedman.statistic:.4f}, p_value {rwt_friedman.pvalue:.10f}')
    print(f'Friedman test for MAE: statistic {rst_friedman.statistic:.4f}, p_value {rst_friedman.pvalue:.10f}')
    print()

    if rwt_friedman.pvalue < 0.01 and rst_friedman.pvalue < 0.01:
        print('At least two groups are significantly different')
    else:
        print('We cannot discard the null H0: No significant difference between groups')
    # dict_heart_part = {0: 'PW', 1: 'LVID', 2: 'IVS'}
    # for i in range(3):
    #     print(dict_heart_part[i])
    #     mpe_friedman = friedmanchisquare(*[mpe[trial][:, i] for trial in args.trials])
    #     mae_friedman = friedmanchisquare(*[mae[trial][:, i] for trial in args.trials])
    #     print(f'Friedman test for MPE: statistic {mpe_friedman.statistic:.4f}, p_value {mpe_friedman.pvalue:.10f}')
    #     print(f'Friedman test for MAE: statistic {mae_friedman.statistic:.4f}, p_value {mae_friedman.pvalue:.10f}')
    #     print()

    ## compute the post hoc test
    mpe_baseline = experiments['trial_1']['mpe']
    mae_baseline = experiments['trial_1']['mae']
    rwt_baseline = experiments['trial_1']['rwt_error']
    rst_baseline = experiments['trial_1']['rst_error']

    violin_dict = {'trial_1': ['Baseline', mpe_baseline, mae_baseline, rwt_baseline, rst_baseline]}
    for trial in args.trials[1:]:
        print(f'Post hoc test for {trial}')
        mpe_trial = experiments[trial]['mpe']
        mae_trial = experiments[trial]['mae']
        rwt_trial = experiments[trial]['rwt_error']
        rst_trial = experiments[trial]['rst_error']

        rwt_wilcoxon = wilcoxon(rwt_baseline, rwt_trial)
        rst_wilcoxon = wilcoxon(rst_baseline, rst_trial)
        print(f'Wilcoxon test for RWT: statistic {rwt_wilcoxon.statistic:.4f}, p_value {rwt_wilcoxon.pvalue:.10f}')
        print(f'Wilcoxon test for RST: statistic {rst_wilcoxon.statistic:.4f}, p_value {rst_wilcoxon.pvalue:.10f}')
        if rwt_wilcoxon.pvalue < 0.01:
            print('!!! RWT Significant !!!')
        if rst_wilcoxon.pvalue < 0.01:
            print('!!! RST Significant !!!')
        violin_dict[trial] = [trial, mpe_trial, mae_trial, rwt_trial, rst_trial]



def boxplot_fold(args, experiments):
    """
    PLot the boxplot of each experiment fold
    """
    boxplot_dict = {}
    for trial in args.trials:
        print(trial)
        mpe = experiments[trial]['mpe_fold']
        mae = experiments[trial]['mae_fold']
        rwt_error = experiments[trial]['rwt_error_fold']
        rst_error = experiments[trial]['rst_error_fold']

        rwt_means = [np.mean(rwt_error[i]) for i in range(len(rwt_error))]
        rst_means = [np.mean(rst_error[i]) for i in range(len(rst_error))]

        pw_means = [np.mean(mpe[i][:, 0]*100) for i in range(len(mae))]
        lvid_means = [np.mean(mpe[i][:, 1]*100) for i in range(len(mae))]
        ivs_means = [np.mean(mpe[i][:, 2]*100) for i in range(len(mae))]

        boxplot_dict[trial] = {'pw': pw_means, 'lvid': lvid_means, 'ivs': ivs_means, 'rwt': rwt_means, 'rst': rst_means}

    fig, ax = plt.subplots(3, 1, figsize=(8,15), tight_layout=True, num='Aug_exp - MPE error')
    ## plot the orizontal value of the medial of trial_1
    ax[0].hlines(np.median(boxplot_dict['trial_1']['pw']), 1, 5, color='darkblue', label='Real')
    ax[1].hlines(np.median(boxplot_dict['trial_1']['lvid']), 1, 5, color='darkblue', label='Real')
    ax[2].hlines(np.median(boxplot_dict['trial_1']['ivs']), 1, 5, color='darkblue', label='Real')

    #set title
    ax[0].set_title('PW', fontsize=26)
    ax[1].set_title('LVID', fontsize=26)
    ax[2].set_title('IVS', fontsize=26)

    ## fillbetween the 25% and 75% of the boxplot of trial_1
    ax[0].fill_between([1, 5], np.percentile(boxplot_dict['trial_1']['pw'], 25), np.percentile(boxplot_dict['trial_1']['pw'], 75), color='lightblue', alpha=0.35)
    ax[1].fill_between([1, 5], np.percentile(boxplot_dict['trial_1']['lvid'], 25), np.percentile(boxplot_dict['trial_1']['lvid'], 75), color='lightblue', alpha=0.35)
    ax[2].fill_between([1, 5], np.percentile(boxplot_dict['trial_1']['ivs'], 25), np.percentile(boxplot_dict['trial_1']['ivs'], 75), color='lightblue', alpha=0.35)

    medians_pw, first_quantile_pw, third_quantile_pw = [], [], []
    medians_lvid, first_quantile_lvid, third_quantile_lvid = [], [], []
    medians_ivs, first_quantile_ivs, third_quantile_ivs = [], [], []
    trial_name = {0.0: 'Real', 0.2: "Real+20%Gen", 0.4: "Real+40%Gen", 0.6: "Real+60%Gen", 0.8: "Real+80%Gen", 1.0: "Real+100%Gen"}
    for i, trial in enumerate(args.trials[1:]):
        name_experiment = trial_name[experiments[trial]['percentace']]
        medians_pw.append(np.median(boxplot_dict[trial]['pw']))
        first_quantile_pw.append(np.percentile(boxplot_dict[trial]['pw'], 25))
        third_quantile_pw.append(np.percentile(boxplot_dict[trial]['pw'], 75))

        medians_lvid.append(np.median(boxplot_dict[trial]['lvid']))
        first_quantile_lvid.append(np.percentile(boxplot_dict[trial]['lvid'], 25))
        third_quantile_lvid.append(np.percentile(boxplot_dict[trial]['lvid'], 75))

        medians_ivs.append(np.median(boxplot_dict[trial]['ivs']))
        first_quantile_ivs.append(np.percentile(boxplot_dict[trial]['ivs'], 25))
        third_quantile_ivs.append(np.percentile(boxplot_dict[trial]['ivs'], 75))


    ax[0].plot(range(1, 6), medians_pw, color='green', marker='o', ms=7, label=r'Real + % genereted data')
    ax[0].fill_between(range(1, 6), first_quantile_pw, third_quantile_pw, color='lightgreen', alpha=0.35)
    ax[1].plot(range(1, 6), medians_lvid, color='green', marker='o', ms=7)
    ax[1].fill_between(range(1, 6), first_quantile_lvid, third_quantile_lvid, color='lightgreen', alpha=0.35)
    ax[2].plot(range(1, 6), medians_ivs, color='green',  marker='o', ms=7)
    ax[2].fill_between(range(1, 6), first_quantile_ivs, third_quantile_ivs, color='lightgreen', alpha=0.35)

    name_ticks = ['20', '40', '60', '80', '100']
    name_ticks_empty = ['']*5

    ax[0].set_xticks(range(1, 6))
    ax[0].set_xticklabels(name_ticks,fontsize=24)
    ax[1].set_xticks(range(1, 6))
    ax[1].set_xticklabels(name_ticks, fontsize=24)


    ax[2].set_xticks(range(1, 6))
    ax[2].set_xticklabels(name_ticks, fontsize=24)

    ## set the tick font size of y axis
    for ax_ in ax:
        ax_.tick_params(axis='both', which='major', labelsize=22)

    ax[0].set_ylabel('MPE [%]', fontsize=24)
    ax[1].set_ylabel('MPE [%]', fontsize=24)
    ax[2].set_ylabel('MPE [%]', fontsize=24)
    ax[2].set_xlabel('Percentage of generated dataset [%]', fontsize=24)

    # set the legend
    # ax[0].legend(fontsize=20)

    # ## set the lim of the y axis
    ax[0].set_ylim(12, 22.5)
    ax[1].set_ylim(5.4, 6.4)
    ax[2].set_ylim(12.5, 17.0)
    # set the grid
    ax[0].grid(linestyle=':')
    ax[1].grid(linestyle=':')
    ax[2].grid(linestyle=':')

    # set the y ticks with single decimal value
    ax[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1f}'))
    ax[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.2f}'))
    ax[2].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1f}')) 




    fig1, ax1 = plt.subplots(2, 1, figsize=(8,15), num='Aug_exp - Relative distances', tight_layout=True)
    ## plot the orizontal value of the medial of trial_1
    ax1[1].hlines(np.median(boxplot_dict['trial_1']['rwt']), 1, 5, color='darkblue', label='Real')
    ax1[0].hlines(np.median(boxplot_dict['trial_1']['rst']), 1, 5, color='darkblue', label='Real')

    #set title
    ax1[0].set_title('RST', fontsize=26)
    ax1[1].set_title('RWT', fontsize=26)

    ## fillbetween the 25% and 75% of the boxplot of trial_1
    ax1[1].fill_between([1, 5], np.percentile(boxplot_dict['trial_1']['rwt'], 25), np.percentile(boxplot_dict['trial_1']['rwt'], 75), color='lightblue', alpha=0.35)
    ax1[0].fill_between([1, 5], np.percentile(boxplot_dict['trial_1']['rst'], 25), np.percentile(boxplot_dict['trial_1']['rst'], 75), color='lightblue', alpha=0.35)

    medians_rwt, first_quantile_rwt, third_quantile_rwt = [], [], []
    medians_rst, first_quantile_rst, third_quantile_rst = [], [], []
    for i, trial in enumerate(args.trials[1:]):
        name_experiment = trial_name[experiments[trial]['percentace']]
        medians_rwt.append(np.median(boxplot_dict[trial]['rwt']))
        first_quantile_rwt.append(np.percentile(boxplot_dict[trial]['rwt'], 25))
        third_quantile_rwt.append(np.percentile(boxplot_dict[trial]['rwt'], 75))

        medians_rst.append(np.median(boxplot_dict[trial]['rst']))
        first_quantile_rst.append(np.percentile(boxplot_dict[trial]['rst'], 25))
        third_quantile_rst.append(np.percentile(boxplot_dict[trial]['rst'], 75))

    ax1[1].plot(range(1, 6), medians_rwt, color='green',  marker='o', ms=7)
    ax1[1].fill_between(range(1, 6), first_quantile_rwt, third_quantile_rwt, color='lightgreen', alpha=0.35)
    ax1[0].plot(range(1, 6), medians_rst, color='green',  marker='o', ms=7,  label=r'Real + % genereted data')
    ax1[0].fill_between(range(1, 6), first_quantile_rst, third_quantile_rst, color='lightgreen', alpha=0.35)

    ax1[0].set_xticks(range(1, 6))
    ax1[0].set_xticklabels(name_ticks, fontsize=24)

    ax1[1].set_xticks(range(1, 6))
    ax1[1].set_xticklabels(name_ticks, fontsize=24)

    ## set the tick font size of y axis
    for ax_ in ax1:
        ax_.tick_params(axis='both', which='major', labelsize=22)


    ax1[0].set_ylabel('MAE [adm]', fontsize=24)
    ax1[1].set_ylabel('MAE [adm]', fontsize=24)

    ## set the lim of the y axis
    ax1[0].set_ylim(0.065, 0.0905)
    ax1[1].set_ylim(0.065, 0.105)

    ax1[1].set_xlabel('Percentage of generated dataset [%]', fontsize=24)

    # set the legend
    ax1[0].legend(fontsize=24)

    # set the grid
    ax1[0].grid(linestyle=':')
    ax1[1].grid(linestyle=':')


    plt.show()





def convergence_curve(args, experiments):
    """
    Plot the convergence curve of each metric
    """
    convergens_curves = {}
    for trial in args.trials:
        print(trial)
        mpe_pw, mpe_lvid, mpe_ivs = experiments[trial]['mpe'][:,0], experiments[trial]['mpe'][:,1], experiments[trial]['mpe'][:,2]
        mae_pw, mae_lvid, mae_ivs = experiments[trial]['mae'][:,0], experiments[trial]['mae'][:,1], experiments[trial]['mae'][:,2]
        rwt_error = experiments[trial]['rwt_error']
        rst_error = experiments[trial]['rst_error']

        p = experiments[trial]['percentace']
        number_samples = len(mpe_pw)
        error_span = np.linspace(0, 1.0, len(mpe_pw))
        error_relative_span = np.linspace(0, 0.5, len(mpe_pw))
        count_samples_pw = np.array([len(np.where(mae_pw < error)[0]) for error in error_span])
        count_samples_lvid = np.array([len(np.where(mae_lvid < error)[0]) for error in error_span])
        count_samples_ivs = np.array([len(np.where(mae_ivs < error)[0]) for error in error_span])
        count_sample_rwt = np.array([len(np.where(rwt_error < error)[0]) for error in error_relative_span])
        count_sample_rst = np.array([len(np.where(rst_error < error)[0]) for error in error_relative_span])

        convergens_curves[trial] = {'pw': count_samples_pw, 'lvid': count_samples_lvid, 'ivs': count_samples_ivs, 'p': p,
                                    'rwt':count_sample_rwt, 'rst':count_sample_rst ,'number_samples': number_samples}

    ## plot the convergence curve
    fig, ax = plt.subplots(1, 3, figsize=(15, 5), tight_layout=True)
    name_dict = {0.0: 'Real', 0.2: "Real+20%Gen", 0.4: "Real+40%Gen", 0.6: "Real+60%Gen", 0.8: "Real+80%Gen", 1.0: "Real+100%Gen"}
    for trial in args.trials:
        name_experiment = name_dict[convergens_curves[trial]['p']]
        patients = convergens_curves[trial]['number_samples']
        ax[0].plot(error_span*10, convergens_curves[trial]['pw']/patients, label=name_experiment)
        ax[1].plot(error_span*10, convergens_curves[trial]['lvid']/patients, label=name_experiment)
        ax[2].plot(error_span*10, convergens_curves[trial]['ivs']/patients, label=name_experiment)
    ax[0].set_title('PW', fontsize=22)
    ax[1].set_title('LVID', fontsize=22)
    ax[2].set_title('IVS', fontsize=22)

    ax[0].set_xlabel('MAE [mm]', fontsize=22)
    ax[1].set_xlabel('MAE [mm]', fontsize=22)
    ax[2].set_xlabel('MAE [mm]', fontsize=22)

    ax[0].set_ylabel('Fraction of patients', fontsize=22)
    ax[0].legend(fontsize=22)
    ax[1].legend(fontsize=22)
    ax[2].legend(fontsize=22)

    ## set the ticks font size
    for ax_ in ax:
        ax_.tick_params(axis='both', which='major', labelsize=20)


    fig1, ax1 = plt.subplots(1, 2, num='Relative distances', figsize=(10, 5), tight_layout=True)
    for trial in args.trials:
        name_experiment = name_dict[convergens_curves[trial]['p']]
        patients = convergens_curves[trial]['number_samples']
        ax1[0].plot(error_relative_span, convergens_curves[trial]['rwt']/patients, label=name_experiment)
        ax1[1].plot(error_relative_span, convergens_curves[trial]['rst']/patients, label=name_experiment)

    ax1[0].set_title('RWT', fontsize=22)
    ax1[1].set_title('RST', fontsize=22)
    ax1[0].set_xlabel('MAE [adm]', fontsize=22)
    ax1[1].set_xlabel('MAE [adm]', fontsize=22)

    ax1[0].set_ylabel('Fraction of patients', fontsize=22)

    ax1[0].legend(fontsize=22)
    ax1[1].legend(fontsize=22)

    ## set the ticks font size
    for ax_ in ax1:
        ax_.tick_params(axis='both', which='major', labelsize=20)

    plt.show()








def main(args):
    """
    main of statistical analysis of RWT and RST
    """
    experiments = {}
    for trial in args.trials:
        ## read the single trial
        single_exp_dict = read_single_trial(args.model_path, trial)
        experiments[trial] = single_exp_dict


    ## compute the Friedman test
    # friedman_and_post_hoc(args, experiments)

    ## plot the boxplot of each fold
    boxplot_fold(args, experiments)

    ## plot the convergence curve
    # convergence_curve(args, experiments)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Read the prediction and the real value of RWT and RST for the statistical analysis')
    parser.add_argument('--model_path', type=str, default="/media/angelo/OS/Users/lasal/OneDrive - Scuola Superiore Sant'Anna/PhD_notes/Echocardiografy/trained_model/regression/Batch2/data_augumentation", help='Directory of models, i.e. trained_model')
    parser.add_argument('--trials', nargs='+', type=str, default=['trial_1'], help='trial number to analyse')

    args = parser.parse_args()

    main(args)