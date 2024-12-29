import pandas as pd
import json
import matplotlib.pyplot as plt
import numpy as np

original_data_x = pd.DataFrame(np.load('data/diabetes/X_num_train.npy'))
original_data_y = pd.DataFrame(np.load('data/diabetes/y_train.npy'))
original_data = pd.concat([pd.DataFrame(original_data_x), pd.DataFrame(original_data_y)], ignore_index=True, axis=1)
dp_x = pd.DataFrame(np.load("exp/diabetes/ddpm_cb_best/X_num_train.npy"))
dp_y = pd.DataFrame(np.load("exp/diabetes/ddpm_cb_best/y_train.npy"))
synthetic_data = pd.concat([pd.DataFrame(dp_x), pd.DataFrame(dp_y)], ignore_index=True, axis=1)

i = 0
plt.hist(original_data[i], bins=50, color = 'green', label='original')
plt.hist(synthetic_data[i], bins=50, color = 'red', label='synthetic')
plt.xlabel('No DP')
plt.legend()
plt.show()