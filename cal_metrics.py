# 脚本 mudule2.py 中
import sys
# sys.path.append("synprivutil") # 将所在的文件夹路径放入sys.path中
sys.path.append("tab-ddpm/scripts")
import pandas as pd
import numpy as np
import json
from privacy_utility_framework.privacy_utility_framework.metrics.privacy_metrics import privacy_metric_calculator, privacy_metric_manager
from privacy_utility_framework.privacy_utility_framework.metrics.utility_metrics import utility_metric_calculator, utility_metric_manager
from privacy_utility_framework.privacy_utility_framework.metrics.privacy_metrics.distance import adversarial_accuracy_class, dcr_class, disco, nndr_class
from privacy_utility_framework.privacy_utility_framework.metrics.privacy_metrics.attacks import inference_class, linkability_class, singlingout_class
from privacy_utility_framework.privacy_utility_framework.metrics.utility_metrics.statistical import basic_stats, correlation, js_similarity, ks_test, mutual_information\
    , wasserstein
import argparse
import os
import warnings
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, help='choose the dataset')
parser.add_argument('--model', type=str, help='choose the model')
args = parser.parse_args()
print(f"dataset is {args.dataset}")
print(f"model is {args.model}")

'''
    Calculate privacy metrics and utility metrics
'''

# original data
# original_data = pd.read_csv("synprivutil/datasets/original/diabetes.csv")
# synthetic_data = pd.read_csv("synprivutil/datasets/synthetic/diabetes_datasets/random_sample.csv")

# read synthetic data and original data from tddpm
# -------shiyu adds
# original data

# num data
o_x_train = pd.DataFrame(np.load(f'data/{args.dataset}/X_num_train.npy',allow_pickle=True))
o_x_val = pd.DataFrame(np.load(f'data/{args.dataset}/X_num_val.npy',allow_pickle=True))
o_x_test = pd.DataFrame(np.load(f'data/{args.dataset}/X_num_test.npy',allow_pickle=True))
o_y_train = pd.DataFrame(np.load(f'data/{args.dataset}/y_train.npy',allow_pickle=True))
o_y_val = pd.DataFrame(np.load(f'data/{args.dataset}/y_val.npy',allow_pickle=True))
o_y_test = pd.DataFrame(np.load(f'data/{args.dataset}/y_test.npy',allow_pickle=True))

# cat data
if os.path.exists(f'data/{args.dataset}/X_cat_train.npy'):
    o_x_cat_train = pd.DataFrame(np.load(f'data/{args.dataset}/X_cat_train.npy',allow_pickle=True))
    o_x_cat_val = pd.DataFrame(np.load(f'data/{args.dataset}/X_cat_val.npy',allow_pickle=True))
    o_x_cat_test = pd.DataFrame(np.load(f'data/{args.dataset}/X_cat_test.npy',allow_pickle=True))
    
    o_x_train = pd.concat([o_x_train, o_x_cat_train], ignore_index=True, axis=1)
    o_x_val = pd.concat([o_x_val, o_x_cat_val], ignore_index=True, axis=1)
    o_x_test = pd.concat([o_x_test, o_x_cat_test], ignore_index=True, axis=1)

# generate data
# ddpm-cb-best
# s = np.load('exp/diabetes/ddpm_cb_best/X_num_train.npy')
# s_y = np.load('exp/diabetes/ddpm_cb_best/y_train.npy')

# generate data
# num data
s = pd.DataFrame(np.load(f'exp/{args.dataset}/{args.model}/X_num_train.npy',allow_pickle=True))
s_y = pd.DataFrame(np.load(f'exp/{args.dataset}/{args.model}/y_train.npy',allow_pickle=True))

# cat data
if os.path.exists(f'exp/{args.dataset}/{args.model}/X_cat_train.npy'):
    s_cat = pd.DataFrame(np.load(f'exp/{args.dataset}/{args.model}/X_cat_train.npy',allow_pickle=True))
    
    s = pd.concat([s, s_cat], ignore_index=True, axis=1)

o = pd.concat([o_x_train, o_x_val, o_x_test], ignore_index=True, axis=0)
o_y = pd.concat([o_y_train, o_y_val, o_y_test], ignore_index=True, axis=0)
original_data = pd.concat([o, o_y], ignore_index=True, axis=1)
synthetic_data = pd.concat([s, s_y], ignore_index=True, axis=1)
control_orig = pd.concat([o_x_test, o_y_test], ignore_index=True, axis=1)
control_orig.columns = control_orig.columns.map(lambda x:str(x))
# control_orig.columns = ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age','Outcome']
# print(control_orig)
original_data.columns = original_data.columns.map(lambda x:str(x))
print(f"original_data.columns: {original_data.columns}")
# original_data.columns = ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age','Outcome']
# original_data[[original_data.columns]] = original_data.apply(lambda x:str(x), axis='columns', result_type='expand')
synthetic_data.columns = synthetic_data.columns.map(lambda x:str(x))
print(f"synthetic_data.columns: {synthetic_data.columns}")

#transform data types in synthetic data
def convert(source_df, target_df):
    
    assert source_df.columns.all() == target_df.columns.all()
    converted_df = source_df.copy()
    for column in converted_df.columns:
        target_dtype = target_df[column].dtype
        try:
            converted_df[column] = converted_df[column].astype(target_dtype)
        except ValueError as e:
            raise ValueError(f"cannot transform {column} to {target_dtype}: {e}")
    
    return converted_df

synthetic_data = convert(synthetic_data, original_data)
# synthetic_data.columns = ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age','Outcome']
# synthetic_data[[synthetic_data.columns]] = synthetic_data.apply(lambda x:str(x), axis='columns', result_type='expand')
print(f"len of original data is {len(original_data)}")
print(f"len of synthetic data is {len(synthetic_data)}")
# print(type(original_data.loc[0][0]))
# print(type(synthetic_data.loc[0][0]))
if len(original_data) > len(synthetic_data):
    original_data = original_data.loc[:len(synthetic_data)-1][:]
elif len(original_data) < len(synthetic_data):
    synthetic_data = synthetic_data.loc[:len(original_data)-1][:]
print("after adjusting the length of both dataset")
print(f"len of original data is {len(original_data)}")
print(f"len of synthetic data is {len(synthetic_data)}")

# there are some datasets whose number exceeds the cpu load, so just get former 1000
original_data = original_data.loc[:999][:]
synthetic_data = synthetic_data.loc[:999][:]
print("The number exceeds the cpu load, so just get former 1000")
print(f"len of original data is {len(original_data)}")
print(f"len of synthetic data is {len(synthetic_data)}")

original_name = args.dataset
synthetic_name = args.model

# Initialize PrivacyMetricManager
p = privacy_metric_manager.PrivacyMetricManager()

metric_p_list = \
    [
        dcr_class.DCRCalculator(original_data, synthetic_data, original_name=original_name, synthetic_name=synthetic_name),
        nndr_class.NNDRCalculator(original_data, synthetic_data, original_name=original_name, synthetic_name=synthetic_name),
        adversarial_accuracy_class.AdversarialAccuracyCalculator(original_data, synthetic_data, original_name=original_name,
                                      synthetic_name=synthetic_name),
        adversarial_accuracy_class.AdversarialAccuracyCalculator_NN(original_data, synthetic_data, original_name=original_name,
                                      synthetic_name=synthetic_name),
        # keys = [’Age’, ’BMI’, ’DiabetesPedigreeFunction’, ’Glucose’, ’BloodPressure’], target = 'Outcome'
        # disco.DisclosureCalculator(original_data, synthetic_data, original_name=original_name,
        #                               synthetic_name=synthetic_name, target='Outcome', keys=['Glucose','BloodPressure','BMI','DiabetesPedigreeFunction','Age']),
        # inference_class.InferenceCalculator(original_data, synthetic_data, original_name=original_name,
        #                                       synthetic_name=synthetic_name, aux_cols=['Pregnancies', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'], secret='Glucose', n_attacks=100, control=control_orig),
        # linkability_class.LinkabilityCalculator(original_data, synthetic_data, original_name=original_name,
        #                               synthetic_name=synthetic_name, aux_cols=(['DiabetesPedigreeFunction', 'Age'], ['BMI', 'Glucose', 'BloodPressure', 'Pregnancies']), n_attacks=200, control=control_orig),
        # singlingout_class.SinglingOutCalculator(original_data, synthetic_data, original_name=original_name,
        #                               synthetic_name=synthetic_name)
    ]
p.add_metric(metric_p_list)
results_p = p.evaluate_all()

'''--------------------------------------------------------------------------------------------------------'''
# Initialize UtilityMetricManager
u = utility_metric_manager.UtilityMetricManager()

#Define metrics to evaluate
metric_u_list = [
    basic_stats.BasicStatsCalculator(original_data, synthetic_data, original_name=original_name, synthetic_name=synthetic_name),
    mutual_information.MICalculator(original_data, synthetic_data, original_name=original_name, synthetic_name=synthetic_name),
    # # default method = 'pearson'
    correlation.CorrelationCalculator(original_data, synthetic_data, original_name=original_name, synthetic_name=synthetic_name),
    js_similarity.JSCalculator(original_data, synthetic_data, original_name=original_name, synthetic_name=synthetic_name),
    ks_test.KSCalculator(original_data, synthetic_data, original_name=original_name, synthetic_name=synthetic_name),
    wasserstein.WassersteinCalculator(original_data, synthetic_data, original_name=original_name, synthetic_name=synthetic_name)
]

# Add metrics to manager and evaluate
u.add_metric(metric_u_list)
results_u = u.evaluate_all()

# Merge two dictionaries
def Merge(dict1, dict2):
    res = {**dict1, **dict2}
    return res

results = Merge(results_p, results_u)
dict = {}
file = open("metrics_benchmark.txt", "a")
for key, value in results.items():
    dict[key] = value
    # Print results
    print(f"{key}: {value}")
    # file = open(f"metrics/{args.dataset}/metrics_{args.model}_dp_old.txt", "a")
    file.write(f"{key}:{value}")
    file.write("\n")
file.write("\n")
file.close()

# # compute SPEARMAN
# corr_s = correlation.CorrelationCalculator(original_data, synthetic_data, original_name=original_name, synthetic_name=synthetic_name)
# result = corr_s.evaluate(method=correlation.CorrelationMethod.SPEARMAN)
# dict = {}
# dict["CorrelationCalculator('Diabetes', 'TVAE'), method = 'spearman'"] = result
# print(dict)

# with open("metrics.json","w") as f:
#     json.dump(dict,f)
#     print("Computing metrics are done.")

# print(results.items())
# key, value = results.items()
# file = open("metrics.txt", "a")
# file.write(f"{dict}")
# file.write("\n")
# file.close()
