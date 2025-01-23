#! /bin/bash
# #tvae: train
# python CTGAN/pipeline_tvae.py --config exp/abalone/tvae/config.toml --train --sample --eval
# python CTGAN/pipeline_tvae.py --config exp/adult/tvae/config.toml --train --sample --eval
# python CTGAN/pipeline_tvae.py --config exp/buddy/tvae/config.toml --train --sample --eval
# python CTGAN/pipeline_tvae.py --config exp/california/tvae/config.toml --train --sample --eval
# python CTGAN/pipeline_tvae.py --config exp/cardio/tvae/config.toml --train --sample --eval
# python CTGAN/pipeline_tvae.py --config exp/churn2/tvae/config.toml --train --sample --eval
# python CTGAN/pipeline_tvae.py --config exp/default/tvae/config.toml --train --sample --eval
# python CTGAN/pipeline_tvae.py --config exp/diabetes/tvae/config.toml --train --sample --eval
# python CTGAN/pipeline_tvae.py --config exp/fb-comments/tvae/config.toml --train --sample --eval
# python CTGAN/pipeline_tvae.py --config exp/gesture/tvae/config.toml --train --sample --eval
# python CTGAN/pipeline_tvae.py --config exp/higgs-small/tvae/config.toml --train --sample --eval
# python CTGAN/pipeline_tvae.py --config exp/house/tvae/config.toml --train --sample --eval
# python CTGAN/pipeline_tvae.py --config exp/insurance/tvae/config.toml --train --sample --eval
# python CTGAN/pipeline_tvae.py --config exp/king/tvae/config.toml --train --sample --eval
# python CTGAN/pipeline_tvae.py --config exp/miniboone/tvae/config.toml --train --sample --eval
# python CTGAN/pipeline_tvae.py --config exp/wilt/tvae/config.toml --train --sample --eval

#tvae: calculate metrics
python cal_metrics.py --dataset abalone --model tvae
python cal_metrics.py --dataset adult --model tvae
python cal_metrics.py --dataset buddy --model tvae
python cal_metrics.py --dataset california --model tvae
python cal_metrics.py --dataset cardio --model tvae
python cal_metrics.py --dataset churn2 --model tvae
python cal_metrics.py --dataset diabetes --model tvae
python cal_metrics.py --dataset fb-comments --model tvae
python cal_metrics.py --dataset gesture --model tvae
python cal_metrics.py --dataset higgs-small --model tvae
python cal_metrics.py --dataset house --model tvae
python cal_metrics.py --dataset insurance --model tvae
python cal_metrics.py --dataset king --model tvae
python cal_metrics.py --dataset miniboone --model tvae
python cal_metrics.py --dataset wilt --model tvae