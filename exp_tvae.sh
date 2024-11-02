#! /bin/bash
#tvae
python CTGAN/pipeline_tvae.py --config exp/abalone/tvae/config.toml --train --sample --eval
python CTGAN/pipeline_tvae.py --config exp/adult/tvae/config.toml --train --sample --eval
python CTGAN/pipeline_tvae.py --config exp/buddy/tvae/config.toml --train --sample --eval
python CTGAN/pipeline_tvae.py --config exp/california/tvae/config.toml --train --sample --eval
python CTGAN/pipeline_tvae.py --config exp/cardio/tvae/config.toml --train --sample --eval
python CTGAN/pipeline_tvae.py --config exp/churn2/tvae/config.toml --train --sample --eval
python CTGAN/pipeline_tvae.py --config exp/default/tvae/config.toml --train --sample --eval
python CTGAN/pipeline_tvae.py --config exp/diabetes/tvae/config.toml --train --sample --eval
python CTGAN/pipeline_tvae.py --config exp/fb-comments/tvae/config.toml --train --sample --eval
python CTGAN/pipeline_tvae.py --config exp/gesture/tvae/config.toml --train --sample --eval
python CTGAN/pipeline_tvae.py --config exp/higgs-small/tvae/config.toml --train --sample --eval
python CTGAN/pipeline_tvae.py --config exp/house/tvae/config.toml --train --sample --eval
python CTGAN/pipeline_tvae.py --config exp/insurance/tvae/config.toml --train --sample --eval
python CTGAN/pipeline_tvae.py --config exp/king/tvae/config.toml --train --sample --eval
python CTGAN/pipeline_tvae.py --config exp/miniboone/tvae/config.toml --train --sample --eval
python CTGAN/pipeline_tvae.py --config exp/wilt/tvae/config.toml --train --sample --eval