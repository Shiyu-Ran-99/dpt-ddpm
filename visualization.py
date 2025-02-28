import pandas as pd
import json
import matplotlib.pyplot as plt
import numpy as np

dataset = 'wilt'
# visualize numerical feature distribution
original_data_x = pd.DataFrame(np.load(f'data/{dataset}/X_num_train.npy'))
original_data_y = pd.DataFrame(np.load(f'data/{dataset}/y_train.npy'))
original_data = pd.concat([original_data_x, original_data_y], ignore_index=True, axis=1)
dp_x = pd.DataFrame(np.load(f"/Users/pigr/Desktop/uzh论文/pythonProject/tab-ddpm/exp/{dataset}/ddpm_cb_best/X_num_train.npy"))
dp_y = pd.DataFrame(np.load(f"/Users/pigr/Desktop/uzh论文/pythonProject/tab-ddpm/exp/{dataset}/ddpm_cb_best/y_train.npy"))
# dp_x = pd.DataFrame(np.load("exp/diabetes/ddpm_cb_best/X_num_train.npy"))
# dp_y = pd.DataFrame(np.load("exp/diabetes/ddpm_cb_best/y_train.npy"))
synthetic_data = pd.concat([dp_x, dp_y], ignore_index=True, axis=1)

synthetic_data = synthetic_data.loc[:len(original_data)-1][:]

i = 0
plt.hist(original_data[i], bins=50, color = 'green', label='original')
plt.hist(synthetic_data[i], bins=50, color = 'red', label='synthetic')
plt.xlabel('DP')
plt.legend()
plt.show()

# visualize categorical feature distribution
original_data_x = pd.DataFrame(np.load(f'data/{dataset}/X_cat_train.npy',allow_pickle=True))
# original_data_y = pd.DataFrame(np.load('data/insurance/y_train.npy'))
# original_data = pd.concat([original_data_x, original_data_y], ignore_index=True, axis=1)
dp_x = pd.DataFrame(np.load(f"/Users/pigr/Desktop/uzh论文/pythonProject/tab-ddpm/exp/{dataset}/ddpm_cb_best/X_cat_train.npy",allow_pickle=True))
# dp_y = pd.DataFrame(np.load("/Users/pigr/Desktop/uzh论文/pythonProject/tab-ddpm/exp/insurance/ddpm_cb_best/y_train.npy"))
# dp_x = pd.DataFrame(np.load("exp/diabetes/ddpm_cb_best/X_num_train.npy"))
# dp_y = pd.DataFrame(np.load("exp/diabetes/ddpm_cb_best/y_train.npy"))
# synthetic_data = pd.concat([dp_x, dp_y], ignore_index=True, axis=1)

dp_x = dp_x.loc[:len(original_data_x)-1][:]
i = 0
plt.hist(original_data_x[i][:len(dp_x[i])], bins=50, color = 'green', label='original',align='left')
plt.hist(dp_x[i], bins=50, color = 'red', label='synthetic',align='right')
plt.xlabel('DP')
plt.legend()
plt.show()