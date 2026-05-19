# Representation-Coherence-guided Denoising for graph collaborative filtering

PyTorch Implementation for NDL: representation-coherence-guided denoising for graph collaborative filtering

--- Based on “DAR: Dimension-Adaptive Recommendation with Multi-Granular Noise Control, SIGIR 2025”, https://github.com/Riwei-HEU/DAR

## Environment Requirements

The code has been tested running under Python 3.10.9. The required packages are as follows:

- pytorch == 2.3.1
- numpy == 2.2.6
- scipy == 1.15.3
- scikit-learn == 1.7.2
- prettytable == 3.17.0


## Quick start

Run one of the prepared configurations:

```bash
bash run.sh movielens 0
bash run.sh yelp 0
bash run.sh book 0
```

The second argument is the GPU id. For a short smoke test:

```bash
EPOCHS=20 bash run.sh movielens 0
```

## Manual run

```bash
python main_ndl.py \
  --dataset movielens \
  --data_path data/ \
  --gpu_id 0 \
  --epoch 500 \
  --context_hops 2 \
  --dim 32 \
  --tau_coh 0.1 \
  --tau_loss 5.0 \
  --ema_decay 0.9 \
  --min_activity 50.0 \
  --warmup_epochs 10 \
  --cl_lambda 0.001 \
  --cl_temp 0.2 \
  --cl_drop 0.1
```

Set `--cl_lambda 0` to disable the contrastive term.

## Main configurations

| Dataset | `tau_coh` | `tau_loss` | `ema_decay` | `min_activity` | `cl_lambda` |
| --- | ---: | ---: | ---: | ---: | ---: |
| MovieLens | 0.1 | 5.0 | 0.9 | 50.0 | 0.001 |
| Yelp | 0.5 | 10.0 | 0.9 | 30.0 | 0.001 |
| Book | 0.5 | 5.0 | 0.5 | 25.0 | 0.001 |

Training reports Recall and NDCG at `[5, 10, 20, 50, 100]`. Logs are written to `logs/`.
