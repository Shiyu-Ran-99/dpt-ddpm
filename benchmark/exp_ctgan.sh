#! /bin/bash
# #ctgan: train
python CTGAN/pipeline_ctgan.py --config exp/abalone/ctgan/config.toml --train --sample --eval
python CTGAN/pipeline_ctgan.py --config exp/adult/ctgan/config.toml --train --sample --eval
python CTGAN/pipeline_ctgan.py --config exp/buddy/ctgan/config.toml --train --sample --eval
python CTGAN/pipeline_ctgan.py --config exp/california/ctgan/config.toml --train --sample --eval
python CTGAN/pipeline_ctgan.py --config exp/cardio/ctgan/config.toml --train --sample --eval
python CTGAN/pipeline_ctgan.py --config exp/churn2/ctgan/config.toml --train --sample --eval
python CTGAN/pipeline_ctgan.py --config exp/default/ctgan/config.toml --train --sample --eval
python CTGAN/pipeline_ctgan.py --config exp/diabetes/ctgan/config.toml --train --sample --eval
python CTGAN/pipeline_ctgan.py --config exp/fb-comments/ctgan/config.toml --train --sample --eval
python CTGAN/pipeline_ctgan.py --config exp/gesture/ctgan/config.toml --train --sample --eval
python CTGAN/pipeline_ctgan.py --config exp/higgs-small/ctgan/config.toml --train --sample --eval
python CTGAN/pipeline_ctgan.py --config exp/house/ctgan/config.toml --train --sample --eval
python CTGAN/pipeline_ctgan.py --config exp/insurance/ctgan/config.toml --train --sample --eval
python CTGAN/pipeline_ctgan.py --config exp/king/ctgan/config.toml --train --sample --eval
python CTGAN/pipeline_ctgan.py --config exp/miniboone/ctgan/config.toml --train --sample --eval
python CTGAN/pipeline_ctgan.py --config exp/wilt/ctgan/config.toml --train --sample --eval

#ctgan: calculate metrics
python cal_metrics.py --dataset abalone --model ctgan
python cal_metrics.py --dataset adult --model ctgan
python cal_metrics.py --dataset buddy --model ctgan
python cal_metrics.py --dataset california --model ctgan
python cal_metrics.py --dataset cardio --model ctgan
python cal_metrics.py --dataset churn2 --model ctgan
python cal_metrics.py --dataset diabetes --model ctgan
python cal_metrics.py --dataset fb-comments --model ctgan
python cal_metrics.py --dataset gesture --model ctgan
python cal_metrics.py --dataset higgs-small --model ctgan
python cal_metrics.py --dataset house --model ctgan
python cal_metrics.py --dataset insurance --model ctgan
python cal_metrics.py --dataset king --model ctgan
python cal_metrics.py --dataset miniboone --model ctgan
python cal_metrics.py --dataset wilt --model ctgan