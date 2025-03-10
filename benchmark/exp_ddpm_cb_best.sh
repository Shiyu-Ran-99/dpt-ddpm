#! /bin/bash
#ddpm_cb_best
python scripts/pipeline.py --config exp/abalone/ddpm_cb_best/config.toml --train --sample --eval
python scripts/pipeline.py --config exp/adult/ddpm_cb_best/config.toml --train --sample --eval
python scripts/pipeline.py --config exp/buddy/ddpm_cb_best/config.toml --train --sample --eval
python scripts/pipeline.py --config exp/california/ddpm_cb_best/config.toml --train --sample --eval
python scripts/pipeline.py --config exp/cardio/ddpm_cb_best/config.toml --train --sample --eval
python scripts/pipeline.py --config exp/churn2/ddpm_cb_best/config.toml --train --sample --eval
python scripts/pipeline.py --config exp/default/ddpm_cb_best/config.toml --train --sample --eval
python scripts/pipeline.py --config exp/diabetes/ddpm_cb_best/config.toml --train --sample --eval
python scripts/pipeline.py --config exp/fb-comments/ddpm_cb_best/config.toml --train --sample --eval
python scripts/pipeline.py --config exp/gesture/ddpm_cb_best/config.toml --train --sample --eval
python scripts/pipeline.py --config exp/higgs-small/ddpm_cb_best/config.toml --train --sample --eval
python scripts/pipeline.py --config exp/house/ddpm_cb_best/config.toml --train --sample --eval
python scripts/pipeline.py --config exp/insurance/ddpm_cb_best/config.toml --train --sample --eval
python scripts/pipeline.py --config exp/king/ddpm_cb_best/config.toml --train --sample --eval
python scripts/pipeline.py --config exp/miniboone/ddpm_cb_best/config.toml --train --sample --eval
python scripts/pipeline.py --config exp/wilt/ddpm_cb_best/config.toml --train --sample --eval

#ddpm_cb_best: calculate metrics
python cal_metrics.py --dataset abalone --model ddpm_cb_best
python cal_metrics.py --dataset adult --model ddpm_cb_best
python cal_metrics.py --dataset buddy --model ddpm_cb_best
python cal_metrics.py --dataset california --model ddpm_cb_best
python cal_metrics.py --dataset cardio --model ddpm_cb_best
python cal_metrics.py --dataset churn2 --model ddpm_cb_best
python cal_metrics.py --dataset diabetes --model ddpm_cb_best
python cal_metrics.py --dataset fb-comments --model ddpm_cb_best
python cal_metrics.py --dataset gesture --model ddpm_cb_best
python cal_metrics.py --dataset higgs-small --model ddpm_cb_best
python cal_metrics.py --dataset house --model ddpm_cb_best
python cal_metrics.py --dataset insurance --model ddpm_cb_best
python cal_metrics.py --dataset king --model ddpm_cb_best
python cal_metrics.py --dataset miniboone --model ddpm_cb_best
python cal_metrics.py --dataset wilt --model ddpm_cb_best
