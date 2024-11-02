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