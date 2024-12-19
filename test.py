# import pickle
# fr = open("synprivutil/synthetic_models/diabetes_models/tvae_model.pkl", 'rb')
# inf = pickle.load(fr)
# print(inf.__dict__)
import numpy as np
import pandas as pd
import pickle
# import lib
tmp = np.load("exp/diabetes/ddpm_cb_best/X_num_train.npy")
print(f"generated data is {tmp[:5, :]}")
print(f"num is {tmp.shape}")
# #
# tmp1 = np.load("data/abalone/X_cat_train.npy", allow_pickle=True)
# # tmp2 = np.load("data/cardio/X_cat_train.npy")
# print("-------------")
# print(f"cat is {tmp1.shape}")
# print(tmp2)
# # import os
# real_data_path = 'data/diabetes/'
# X_num_train, X_cat_train, y_train = lib.read_pure_data(real_data_path, 'train')
#
# X = lib.concat_to_pd(X_num_train, X_cat_train, y_train)

# X.columns = [str(_) for _ in X.columns]
# print(f"fit data X is {X.iloc[0]}")
#
# parent_dir = 'exp/diabetes/tvae/'
# device = "cpu"
# with open(parent_dir + "tvae.obj", 'rb') as f:
#     synthesizer = pickle.load(f)
#     synthesizer.decoder = synthesizer.decoder.to(device)
#
# gen_data = synthesizer.sample(5, seed=0)
# print(f"gen_data is {gen_data}")




# from privacy_utility_framework.privacy_utility_framework.metrics.privacy_metrics import privacy_metric_calculator, privacy_metric_manager
# from privacy_utility_framework.privacy_utility_framework.metrics.privacy_metrics.distance import dcr_class
# original_name = "Diabetes"
# synthetic_name = "TVAE"
#
# # Initialize PrivacyMetricManager
# p = privacy_metric_manager.PrivacyMetricManager()
#
# dcr_class.DCRCalculator(original_data, synthetic_data, original_name=original_name, synthetic_name=synthetic_name)
#
# -------------------- verify if the difference results from data-preprocessing
from synprivutil.privacy_utility_framework.privacy_utility_framework.synthesizers.synthesizers import GaussianMixtureModel, \
    GaussianCopulaModel, CTGANModel, CopulaGANModel, TVAEModel, RandomModel
import pandas as pd
tvae_model = TVAEModel.load_model("synprivutil/synthetic_models/diabetes_models/tvae_model.pkl")
data = pd.read_csv("synprivutil/datasets/original/diabetes.csv", delimiter=',')
print("successfully loaded")
# samples_from_loaded_model = tvae_model.sample(768)
samples_from_loaded_model = tvae_model.sample(5)
print(f"~~~~~Samples from loaded TVAE Model~~~~~\n {samples_from_loaded_model}")

