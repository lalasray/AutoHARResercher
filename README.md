# AutoResearcHAR

Multi-agent pipeline search for IMU human activity recognition, based on [AutoResearcHAR](https://doi.org/10.1145/3830727.3834823). PyTorch handles training and GPU augmentation; Codex or OpenCode supplies agent reasoning.

**Note on Table 2:** SDs added during shepherding were mistakenly repeated across the five fixed-model baselines for each dataset and augmentation condition. These entries are incorrect.

Results for at least one seed are available on [Google Drive](https://drive.google.com/drive/folders/1lRr8VazCkrZiaM8vcLrpNOTEsttt5Gox?usp=sharing). See [dataset protocols](examples/DATASET_PROTOCOLS.md) for exact class IDs and held-out groups.

## Setup

Use Python 3.10+, CUDA-enabled PyTorch, and an authenticated Codex CLI or configured OpenCode provider.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[whar,mcp]'
```

## Run

Replace `YOUR_MODEL_ID` below, then start a 10-round search. UCI-HAR is the default target; use `--target DATASET_ID` to select another dataset.

```bash
autohar search --manifest data/prepared-uci-har/manifest.json \
  --protocol loso --target uci_har \
  --backend codex --model YOUR_MODEL_ID \
  --seeds 13 --rounds 10 --budget 80 \
  --device cuda --output artifacts/uci-har-run
```
