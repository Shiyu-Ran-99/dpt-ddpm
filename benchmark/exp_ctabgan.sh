#! /bin/bash
# #ctabgan: train
python CTAB-GAN/pipeline_ctabgan.py --config exp/abalone/ctabgan/config.toml --train --sample --eval
python CTAB-GAN/pipeline_ctabgan.py --config exp/adult/ctabgan/config.toml --train --sample --eval
python CTAB-GAN/pipeline_ctabgan.py --config exp/buddy/ctabgan/config.toml --train --sample --eval
python CTAB-GAN/pipeline_ctabgan.py --config exp/california/ctabgan/config.toml --train --sample --eval
python CTAB-GAN/pipeline_ctabgan.py --config exp/cardio/ctabgan/config.toml --train --sample --eval
python CTAB-GAN/pipeline_ctabgan.py --config exp/churn2/ctabgan/config.toml --train --sample --eval
python CTAB-GAN/pipeline_ctabgan.py --config exp/default/ctabgan/config.toml --train --sample --eval
python CTAB-GAN/pipeline_ctabgan.py --config exp/diabetes/ctabgan/config.toml --train --sample --eval
python CTAB-GAN/pipeline_ctabgan.py --config exp/fb-comments/ctabgan/config.toml --train --sample --eval
python CTAB-GAN/pipeline_ctabgan.py --config exp/gesture/ctabgan/config.toml --train --sample --eval
python CTAB-GAN/pipeline_ctabgan.py --config exp/higgs-small/ctabgan/config.toml --train --sample --eval
python CTAB-GAN/pipeline_ctabgan.py --config exp/house/ctabgan/config.toml --train --sample --eval
python CTAB-GAN/pipeline_ctabgan.py --config exp/insurance/ctabgan/config.toml --train --sample --eval
python CTAB-GAN/pipeline_ctabgan.py --config exp/king/ctabgan/config.toml --train --sample --eval
python CTAB-GAN/pipeline_ctabgan.py --config exp/miniboone/ctabgan/config.toml --train --sample --eval
python CTAB-GAN/pipeline_ctabgan.py --config exp/wilt/ctabgan/config.toml --train --sample --eval

#ctabgan: calculate metrics
python cal_metrics.py --dataset abalone --model ctabgan
python cal_metrics.py --dataset adult --model ctabgan
python cal_metrics.py --dataset buddy --model ctabgan
python cal_metrics.py --dataset california --model ctabgan
python cal_metrics.py --dataset cardio --model ctabgan
python cal_metrics.py --dataset churn2 --model ctabgan
python cal_metrics.py --dataset diabetes --model ctabgan
python cal_metrics.py --dataset fb-comments --model ctabgan
python cal_metrics.py --dataset gesture --model ctabgan
python cal_metrics.py --dataset higgs-small --model ctabgan
python cal_metrics.py --dataset house --model ctabgan
python cal_metrics.py --dataset insurance --model ctabgan
python cal_metrics.py --dataset king --model ctabgan
python cal_metrics.py --dataset miniboone --model ctabgan
python cal_metrics.py --dataset wilt --model ctabgan