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


'''
    Calculate privacy metrics and utility metrics
'''

# original data
# original_data = pd.read_csv("synprivutil/datasets/original/diabetes.csv")
# synthetic_data = pd.read_csv("synprivutil/datasets/synthetic/diabetes_datasets/random_sample.csv")

# read synthetic data and original data from tddpm
# -------shiyu adds
o_x_train = np.load('data/diabetes/X_num_train.npy')
o_x_val = np.load('data/diabetes/X_num_val.npy')
o_x_test = np.load('data/diabetes/X_num_test.npy')
s = np.load('exp/diabetes/ddpm_cb_best/X_num_train.npy')
o_y_train = np.load('data/diabetes/y_train.npy')
o_y_val = np.load('data/diabetes/y_val.npy')
o_y_test = np.load('data/diabetes/y_test.npy')
s_y = np.load('exp/diabetes/ddpm_cb_best/y_train.npy')
o = pd.concat([pd.DataFrame(o_x_train), pd.DataFrame(o_x_val), pd.DataFrame(o_x_test)], ignore_index=True, axis=0)
o_y = pd.concat([pd.DataFrame(o_y_train), pd.DataFrame(o_y_val), pd.DataFrame(o_y_test)], ignore_index=True, axis=0)
original_data = pd.concat([pd.DataFrame(o), pd.DataFrame(o_y)], ignore_index=True, axis=1)
synthetic_data = pd.concat([pd.DataFrame(s), pd.DataFrame(s_y)], ignore_index=True, axis=1)
control_orig = pd.concat([pd.DataFrame(o_x_test), pd.DataFrame(o_y_test)], ignore_index=True, axis=1)
control_orig.columns = control_orig.columns.map(lambda x:str(x))
control_orig.columns = ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age','Outcome']
# print(control_orig)
original_data.columns = original_data.columns.map(lambda x:str(x))
original_data.columns = ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age','Outcome']
# original_data[[original_data.columns]] = original_data.apply(lambda x:str(x), axis='columns', result_type='expand')
synthetic_data.columns = synthetic_data.columns.map(lambda x:str(x))
synthetic_data.columns = ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age','Outcome']
# synthetic_data[[synthetic_data.columns]] = synthetic_data.apply(lambda x:str(x), axis='columns', result_type='expand')
print(len(original_data))
print(len(synthetic_data))
print(synthetic_data.columns)

original_name = "Diabetes"
synthetic_name = "DDPM-CB-BEST"

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
        disco.DisclosureCalculator(original_data, synthetic_data, original_name=original_name,
                                      synthetic_name=synthetic_name, target='Outcome', keys=['Glucose','BloodPressure','BMI','DiabetesPedigreeFunction','Age']),
        inference_class.InferenceCalculator(original_data, synthetic_data, original_name=original_name,
                                              synthetic_name=synthetic_name, aux_cols=['Pregnancies', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'], secret='Glucose', n_attacks=100, control=control_orig),
        linkability_class.LinkabilityCalculator(original_data, synthetic_data, original_name=original_name,
                                      synthetic_name=synthetic_name, aux_cols=(['DiabetesPedigreeFunction', 'Age'], ['BMI', 'Glucose', 'BloodPressure', 'Pregnancies']), n_attacks=200, control=control_orig),
        singlingout_class.SinglingOutCalculator(original_data, synthetic_data, original_name=original_name,
                                      synthetic_name=synthetic_name)
    ]
p.add_metric(metric_p_list)
results_p = p.evaluate_all()

'''--------------------------------------------------------------------------------------------------------'''
# Initialize UtilityMetricManager
u = utility_metric_manager.UtilityMetricManager()

# Define metrics to evaluate
metric_u_list = [
    basic_stats.BasicStatsCalculator(original_data, synthetic_data, original_name=original_name, synthetic_name=synthetic_name),
    mutual_information.MICalculator(original_data, synthetic_data, original_name=original_name, synthetic_name=synthetic_name),
    # default method = 'pearson'
    correlation.CorrelationCalculator(original_data, synthetic_data, original_name=original_name, synthetic_name=synthetic_name),
    js_similarity.JSCalculator(original_data, synthetic_data, original_name=original_name, synthetic_name=synthetic_name),
    ks_test.KSCalculator(original_data, synthetic_data, original_name=original_name, synthetic_name=synthetic_name),
    # wasserstein.WassersteinCalculator(original_data, synthetic_data, original_name=original_name, synthetic_name=synthetic_name)
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
for key, value in results.items():
    dict[key] = value
    # Print results
    print(f"{key}: {value}")

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
file = open("metrics.txt", "a")
file.write(f"{dict}")
file.write("\n")
file.close()
