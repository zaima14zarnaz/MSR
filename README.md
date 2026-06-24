# TSPNet (Temporal Saliency Prediction Network)

Train and evaluate a two-stage saliency ranking model:

1. **Salient region extractor** — `model/tspnet_model/salient_region_extraction_network.py` (`SalientRegionExtractionNetwork`)
2. **Ranking head** — `model/tspnet_model/tspnet.py` (`TSPNet`)

Data loading uses `pre_process/dataloader.py` (`SaliencyDataset`) and `pre_process/collate.py` (`variable_collate_fn`). Losses and metrics live in `model/losses.py` and `model/metrics.py`.

## Setup

From the repository root (`sr-model1/`):

```bash
cd model
```

Use a Python environment with PyTorch, torchvision, torch-geometric, scipy, tqdm, Pillow, and NLTK (the dataloader downloads the `punkt` tokenizer on first run).

## Dataset layout

`model/train.py` and `model/evaluation.py` expect a dataset directory **outside this repo** (default: `../Dataset/IRSR_ASSR` relative to `model/`). You must provide:

```text
Dataset/<name>/
  images/train/          # RGB images
  images/val/
  images/test/
  rank_order/train/      # per-image rank annotations
  rank_order/val/
  rank_order/test/
  obj_seg_data_train.json
  obj_seg_data_val.json
  obj_seg_data_test.json
  train_gpt4v.csv        # training/val descriptions
  test.csv               # test descriptions
```

Update `dataset_dir` at the top of `model/train.py` or `model/evaluation.py` if your data lives elsewhere.

## Training

Edit settings in `model/train.py` as needed:

- `dataset_dir` — path to the dataset root
- `device` — set to your GPU (e.g. `"cuda:0"`) before training
- `num_epochs`, batch size, learning rates, and train/val split inside `train()`

Run training from `model/`:

```bash
python train.py
```

The script:

- Builds loaders with `pre_process/dataloader.py` and `pre_process/collate.py`
- Trains `SalientRegionExtractionNetwork` + `TSPNet` jointly via `run_component_analysis_batch()`
- Computes losses with `model/losses.py` and ranking metrics with `model/metrics.py`
- Saves checkpoints under `../tspnet-checkpoints/<timestamp>/` with keys `bsd_model` and `rank_model`

Each checkpoint is a `.pth` file (not tracked in git). Typical filenames:

- `best_loss_run_0.pth`
- `best_sor_run_0.pth`
- `epoch_<n>_0_<sor>.pth`

## Evaluation

Edit settings in `model/evaluation.py` before running:

- `dataset_dir` — test dataset root (same layout as above)
- `model_dir` — path to a trained checkpoint `.pth` from training
- `device` — GPU/CPU device string
- `confidence_threshold` — ROI filter threshold (0.5 for IRSR/SIFR, 0.4 for ASSR per comments in the script)

Run evaluation from `model/`:

```bash
python evaluation.py
```

The script:

- Loads the test split via `pre_process/dataloader.py`
- Restores weights into `SalientRegionExtractionNetwork` and `TSPNet`
- Runs inference with `run_component_analysis_batch()`
- Reports **SOR**, **SA-SOR**, **MAE**, phrase–rank Spearman ρ, and timing using `model/metrics.py` and `model/losses.py`

## File reference (tracked in this repo)

| Role | Path |
|------|------|
| Training entry point | `model/train.py` |
| Evaluation entry point | `model/evaluation.py` |
| Backbone + mask head | `model/tspnet_model/salient_region_extraction_network.py` |
| GAT + rank head | `model/tspnet_model/tspnet.py` |
| Losses | `model/losses.py` |
| Metrics / ROI filtering | `model/metrics.py` |
| Dataset | `pre_process/dataloader.py` |
| Batch collation | `pre_process/collate.py` |

## Notes

- Checkpoints (`.pth`), generated saliency maps, and local dataset files are not part of the git repository; create or download them before evaluation.
- `model/evaluation_ca.py` is an alternate evaluation script in the repo; the main test workflow above uses `model/evaluation.py`.
