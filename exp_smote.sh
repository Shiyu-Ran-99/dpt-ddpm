#! /bin/bash
#smote
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