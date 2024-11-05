from scripts.resample_privacy import privacy_metrics
from collections import defaultdict
import json
import numpy as np
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, help='choose the dataset')
args = parser.parse_args()

# datasets = ['abalone','adult','buddy','california','cardio','churn2','diabetes','fb-comments','gesture','higgs-small','house','insurance','king','miniboone','wilt']
# datasets = ['abalone','diabetes']
models = ['ddpm_cb_best','ctabgan-plus','tvae','smote']
# pbar = tqdm(datasets)
mbar = tqdm(models)

res = defaultdict(int)
for model in mbar:
    real_path = 'data/'+args.dataset
    fake_path = 'exp/'+args.dataset+'/'+model
    dists = privacy_metrics(real_path=real_path,fake_path=fake_path, data_percent=15)
    privacy_val = np.median(dists)
    res[model] = privacy_val
    mbar.set_description(f"{model} is completed")

with open(f"dcr/{args.dataset}_dcr.json","w") as f:
    json.dump(res,f)
    print("Computing DCR values are done.")
