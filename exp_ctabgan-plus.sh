#! /bin/bash
#ctabgan-plus: train
# python CTAB-GAN-Plus/pipeline_ctabganp.py --config exp/abalone/ctabgan-plus/config.toml --train --sample --eval
# python CTAB-GAN-Plus/pipeline_ctabganp.py --config exp/adult/ctabgan-plus/config.toml --train --sample --eval
# python CTAB-GAN-Plus/pipeline_ctabganp.py --config exp/buddy/ctabgan-plus/config.toml --train --sample --eval
# python CTAB-GAN-Plus/pipeline_ctabganp.py --config exp/california/ctabgan-plus/config.toml --train --sample --eval
# python CTAB-GAN-Plus/pipeline_ctabganp.py --config exp/cardio/ctabgan-plus/config.toml --train --sample --eval
# python CTAB-GAN-Plus/pipeline_ctabganp.py --config exp/churn2/ctabgan-plus/config.toml --train --sample --eval
# python CTAB-GAN-Plus/pipeline_ctabganp.py --config exp/default/ctabgan-plus/config.toml --train --sample --eval
# python CTAB-GAN-Plus/pipeline_ctabganp.py --config exp/diabetes/ctabgan-plus/config.toml --train --sample --eval
# python CTAB-GAN-Plus/pipeline_ctabganp.py --config exp/fb-comments/ctabgan-plus/config.toml --train --sample --eval
# python CTAB-GAN-Plus/pipeline_ctabganp.py --config exp/gesture/ctabgan-plus/config.toml --train --sample --eval
# python CTAB-GAN-Plus/pipeline_ctabganp.py --config exp/higgs-small/ctabgan-plus/config.toml --train --sample --eval
# python CTAB-GAN-Plus/pipeline_ctabganp.py --config exp/house/ctabgan-plus/config.toml --train --sample --eval
# python CTAB-GAN-Plus/pipeline_ctabganp.py --config exp/insurance/ctabgan-plus/config.toml --train --sample --eval
# python CTAB-GAN-Plus/pipeline_ctabganp.py --config exp/king/ctabgan-plus/config.toml --train --sample --eval
# python CTAB-GAN-Plus/pipeline_ctabganp.py --config exp/miniboone/ctabgan-plus/config.toml --train --sample --eval
# python CTAB-GAN-Plus/pipeline_ctabganp.py --config exp/wilt/ctabgan-plus/config.toml --train --sample --eval

#ctabgan-plus: calculate metrics
python cal_metrics.py --dataset abalone --model ctabgan-plus
python cal_metrics.py --dataset adult --model ctabgan-plus
python cal_metrics.py --dataset buddy --model ctabgan-plus
python cal_metrics.py --dataset california --model ctabgan-plus
python cal_metrics.py --dataset cardio --model ctabgan-plus
python cal_metrics.py --dataset churn2 --model ctabgan-plus
python cal_metrics.py --dataset diabetes --model ctabgan-plus
python cal_metrics.py --dataset fb-comments --model ctabgan-plus
python cal_metrics.py --dataset gesture --model ctabgan-plus
python cal_metrics.py --dataset higgs-small --model ctabgan-plus
python cal_metrics.py --dataset house --model ctabgan-plus
python cal_metrics.py --dataset insurance --model ctabgan-plus
python cal_metrics.py --dataset king --model ctabgan-plus
python cal_metrics.py --dataset miniboone --model ctabgan-plus
python cal_metrics.py --dataset wilt --model ctabgan-plus