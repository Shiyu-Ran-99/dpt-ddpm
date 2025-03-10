#! /bin/bash
# #smote: train
python smote/pipeline_smote.py --config exp/abalone/smote/config.toml --sample --eval
python smote/pipeline_smote.py --config exp/adult/smote/config.toml --sample --eval
python smote/pipeline_smote.py --config exp/buddy/smote/config.toml --sample --eval
python smote/pipeline_smote.py --config exp/california/smote/config.toml --sample --eval
python smote/pipeline_smote.py --config exp/cardio/smote/config.toml --sample --eval
python smote/pipeline_smote.py --config exp/churn2/smote/config.toml --sample --eval
python smote/pipeline_smote.py --config exp/default/smote/config.toml --sample --eval
python smote/pipeline_smote.py --config exp/diabetes/smote/config.toml --sample --eval
python smote/pipeline_smote.py --config exp/fb-comments/smote/config.toml --sample --eval
python smote/pipeline_smote.py --config exp/gesture/smote/config.toml --sample --eval
python smote/pipeline_smote.py --config exp/higgs-small/smote/config.toml --sample --eval
python smote/pipeline_smote.py --config exp/house/smote/config.toml --sample --eval
python smote/pipeline_smote.py --config exp/insurance/smote/config.toml --sample --eval
python smote/pipeline_smote.py --config exp/king/smote/config.toml --sample --eval
python smote/pipeline_smote.py --config exp/miniboone/smote/config.toml --sample --eval
python smote/pipeline_smote.py --config exp/wilt/smote/config.toml --sample --eval

#smote: calculate metrics
python cal_metrics.py --dataset abalone --model smote
python cal_metrics.py --dataset adult --model smote
python cal_metrics.py --dataset buddy --model smote
python cal_metrics.py --dataset california --model smote
python cal_metrics.py --dataset cardio --model smote
python cal_metrics.py --dataset churn2 --model smote
python cal_metrics.py --dataset diabetes --model smote
python cal_metrics.py --dataset fb-comments --model smote
python cal_metrics.py --dataset gesture --model smote
python cal_metrics.py --dataset higgs-small --model smote
python cal_metrics.py --dataset house --model smote
python cal_metrics.py --dataset insurance --model smote
python cal_metrics.py --dataset king --model smote
python cal_metrics.py --dataset miniboone --model smote
python cal_metrics.py --dataset wilt --model smote