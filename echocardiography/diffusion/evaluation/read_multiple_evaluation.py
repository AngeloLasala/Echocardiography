"""
Evalutae the statistical difference accross multiple trial
"""
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon, friedmanchisquare
import seaborn as sns

from echocardiography.diffusion.evaluation.read_evaluation import read_hypertrophy_consistency

def friedman_test(data):
	"""
	Perform the friedman test
	"""	
	# Perform the Friedman test
	statistic, p_value = friedmanchisquare(*data)

	print(f"Friedman Test Statistic: {statistic}")
	print(f"P-value: {p_value}")
	if p_value < 0.05:
		print('There is a significant difference between the groups')
		print()
		return True
	else:
		print('There is no significant difference between the groups')
		print()
		return False

def main(args_parser, conds_dict):
	"""
	Main of the multiple analysis
	"""
	experiment_dir = os.path.join(args_parser.par_dir, args_parser.trial)
	
	rwt_list = {}
	rst_list = {}
	for experiment in conds_dict.keys():
		experiment_path = os.path.join(experiment_dir, experiment)
		hypertrophy_path = os.path.join(experiment_path, 'hypertrophy_evaluation')
		eval_dict = read_hypertrophy_consistency(hypertrophy_path)[conds_dict[experiment]['guide_w']][conds_dict[experiment]['epoch']]

		rwt_list[experiment] = (eval_dict['rwt_gen'])
		rst_list[experiment] = (eval_dict['rst_gen'])

	rwt_list['real'] = eval_dict['rwt_real']
	rst_list['real'] = eval_dict['rst_real']
	print(len(rwt_list))
	print(len(rst_list))

	rwt_list = [rwt_list[experiment] for experiment in args_parser.experiments] + [rwt_list['real']]
	rst_list = [rst_list[experiment] for experiment in args_parser.experiments] + [rst_list['real']]

	## compute the friedman test as omnibus test
	# compute the friedman test
	rwt_bool = friedman_test(rwt_list)
	rst_bool = friedman_test(rst_list)

	name_list = [conds_dict[i]['name'] for i in args_parser.experiments] + ['real']
	print(name_list)


	print('RWT')
	if rwt_bool:
		# compute the wilcoxon test
		alpha = 0.05/(len(rwt_list) - 1)
		for i in range(len(rwt_list)-1):
			statistic, p_value = wilcoxon(rwt_list[i], rwt_list[-1])
			print(name_list[i] + ' vs ' + name_list[-1])
			print(f"Wilcoxon Test Statistic  {statistic:.5}, p-value {p_value:.5}")
			if p_value < alpha:print('There is a significant difference between the groups')
			else: print('There is no significant difference between the groups')
			print()
	else:
		print('No Wilcoxon test for rwt_list')
		print()
	
	print("====================================================")

	print('RST')
	if rst_bool:
		# compute the wilcoxon test
		alpha = 0.05/(len(rst_list) - 1)
		for i in range(len(rst_list)-1):
			statistic, p_value = wilcoxon(rst_list[i], rst_list[-1])
			print(name_list[i] + ' vs ' + name_list[-1])
			print(f"Wilcoxon Test Statistic  {statistic:.5}, p-value {p_value:.5}")
			if p_value < alpha:print('There is a significant difference between the groups')
			else: print('There is no significant difference between the groups')
			print()
	else:
		print('No Wilcoxon test for rst_list')
		print()

	
		## create a unique figure with the violin plot using seaborn
	# create the dataframe
	rwt_data = rwt_list.copy()
	rst_data = rst_list.copy()

	for i, j in zip(rwt_data, rst_data):
		median_rwt, first_quantile_rwt, third_quantile_rwt = np.mean(i), np.quantile(i, 0.25), np.quantile(i, 0.75)
		median_rst, first_quantile_rst, third_quantile_rst = np.mean(j), np.quantile(j, 0.25), np.quantile(j, 0.75)
		
	
		outliers_rwt = np.where(i > 5 * third_quantile_rwt)
		outliers_rst = np.where(j > 5 * third_quantile_rst)
		print(outliers_rwt[0].shape)
		# print(outliers_rst[0].shape)
		# print()

		i[outliers_rwt] = median_rwt
		j[outliers_rst] = median_rst



	df_rwt = pd.DataFrame(rwt_data).T
	df_rwt.columns = name_list
	df_rwt = pd.melt(df_rwt, var_name='Condition', value_name='RWT')
	

	df_rst = pd.DataFrame(rst_data).T
	df_rst.columns = name_list
	df_rst = pd.melt(df_rst, var_name='Condition', value_name='RST')

	color_list = ['lightgreen', 'lightgreen', 'lightgreen', 'lightgreen', 'lightgreen', 'lightgreen', 'lightblue']
	# create the violin plot
	plt.figure(figsize=(20, 8),  num='Statistical analysis', tight_layout=True)
	plt.subplot(1, 2, 1)
	sns.violinplot(x='Condition', y='RWT', data=df_rwt, inner=None,  linewidth=1,
						hue='Condition', palette=color_list[:len(df_rwt['Condition'].unique())], legend=False)
	sns.stripplot(x='Condition', y='RWT', data=df_rwt, color='k', alpha=0.5, jitter=True, size=1.5)
	
	if rwt_bool:
		# compute the wilcoxon test
		alpha = 0.05/(len(rwt_list) - 1)
		s=0
		for i in range(len(rwt_list)-1):
			statistic, p_value = wilcoxon(rwt_list[i], rwt_list[-1])
			if p_value < alpha:
				s+=1
				plt.plot([i, 6], [(s/5)+2.2, (s/5)+2.2], color='black', lw=1.5)
				if p_value < 0.001: text = '***'
				if (p_value < 0.01) and (p_value > 0.001): text = '**'
				if (p_value < 0.05) and (p_value > 0.01): text = '*'
				plt.text(i + (6-i)/2, (s/5)+2.2, text, ha='center', fontsize=14)
			else: pass
	plt.xticks(fontsize=18, rotation=30)
	plt.yticks(fontsize=20)
	plt.ylabel('RWT', fontsize=25)
	plt.xlabel('', fontsize=25)
	plt.grid(linestyle=':')
	plt.ylim([-0.1,3.2])

	plt.subplot(1, 2, 2)
	sns.violinplot(x='Condition', y='RST', data=df_rst, inner=None, linewidth=1,
					hue='Condition', palette=color_list[:len(df_rwt['Condition'].unique())], legend=False)
	sns.stripplot(x='Condition', y='RST', data=df_rst, color='k', alpha=0.5, jitter=True, size=1.5)

	if rwt_bool:
		# compute the wilcoxon test
		alpha = 0.05/(len(rst_list) - 1)
		s = 0
		for i in range(len(rst_list)-1):
			statistic, p_value = wilcoxon(rst_list[i], rst_list[-1])
			if p_value < alpha:
				s+=1
				plt.plot([i, 6], [(s/5)+2.2, (s/5)+2.2], color='black', lw=1.5)
				if p_value < 0.001: text = '***'
				if (p_value < 0.01) and (p_value > 0.001): text = '**'
				if (p_value < 0.05) and (p_value > 0.01): text = '*'
				plt.text(i + (6-i)/2, (s/5)+2.2, text, ha='center', fontsize=14)
			else: pass

	## set the font size
	plt.xticks(fontsize=18, rotation=30)
	plt.yticks(fontsize=20)
	plt.ylabel('RST', fontsize=20)
	plt.xlabel('', fontsize=25)
	plt.grid(linestyle=':')
	plt.ylim([-0.1,3.2])
	plt.show()




	


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Evaluate the statistical difference accross multiple trial')
	parser.add_argument('--par_dir', type=str, default='/home/angelo/Documents/Echocardiography/echocardiography/diffusion/trained_model/eco',
						 help="""parent directory of the folder with the evaluation file, it is the same as the trained model
						local: /home/angelo/Documents/Echocardiography/echocardiography/diffusion/trained_model/eco
						cluster: /leonardo_work/IscrC_Med-LMGM/Angelo/trained_model/diffusion/eco""")
	parser.add_argument('--trial', type=str, default='trial_1', help='trial name for saving the model, it is the trial folde that contain the VAE model')
	parser.add_argument('--experiments', nargs='+',  default=['ldm_1', 'cond_ldm_2', 'cond_ldm_4', 'cond_ldm_5', 'cond_ldm_8', 'cond_ldm_1'], help='epoch of the model to evaluate')	

	args = parser.parse_args()

	conds_dict = { 'ldm_1': {'name':'baseline', 'epoch': '60', 'guide_w': '-1.0'},
				   'cond_ldm_2': {'name':'class-cond', 'epoch': '120', 'guide_w': '2.0'},
				   'cond_ldm_4': {'name':'keypoint-cond', 'epoch': '150', 'guide_w': '0.0'},
				   'cond_ldm_1': {'name':'geom-anat-cond', 'epoch': '100', 'guide_w': '0.6'},
				   'cond_ldm_5': {'name':'parameter-cond', 'epoch': '100', 'guide_w': '0.0'},
				   'cond_ldm_8': {'name':'geom-anat', 'epoch': '150', 'guide_w': '0.4'},
	}

	main(args, conds_dict)