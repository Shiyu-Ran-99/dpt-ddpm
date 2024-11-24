import numpy as np
from tqdm import tqdm
from collections import defaultdict
import json
from resample_privacy import privacy_metrics
# t = np.load('data/abalone/X_num_train.npy')
# print(t)
models = ['ddpm_cb_best']
# pbar = tqdm(datasets)
mbar = tqdm(models)

res = defaultdict(int)
for model in mbar:
    real_path = 'data/'+'diabetes'
    fake_path = 'exp/'+'diabetes'+'/'+model
    dists = privacy_metrics(real_path=real_path,fake_path=fake_path, data_percent=15)
    privacy_val = np.median(dists)
    res[model] = privacy_val
    mbar.set_description(f"{model} is completed")

with open("dcr/diabetes_dcr.json","w") as f:
    json.dump(res,f)
    print("Computing DCR values are done.")