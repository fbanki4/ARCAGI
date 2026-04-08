# ViT with VICReg for Abstraction and Reasoning Corpus (ARC)

A Vision Transformer (ViT) model pre-trained with VICReg (Variance-Invariance-Covariance Regularization) self-supervised learning to solve tasks from the [Abstraction and Reasoning Corpus (ARC)](https://github.com/fchollet/ARC-AGI). Supports VICReg pre-training, supervised decoder fine-tuning, and submission file generation.

## Project Structure

```
ARCAGI/
├── arc_solver_vicreg.py                 # Main script — model, training, inference
├── arc-agi_training_challenges.json     # ARC training tasks (input/output grids)
├── arc-agi_training_solutions.json      # Solutions for training tasks
├── requirements.txt                     # Python dependencies
└── README.md
```

## How It Works

The solver follows a three-phase pipeline:

1. **VICReg Pre-training** — Train the ViT encoder with self-supervision by generating two randomly masked views of concatenated input-output grids and enforcing variance, invariance, and covariance constraints on their representations.
2. **Decoder Fine-tuning** — Freeze the pre-trained encoder and train a decoder head to predict output grids from encoded inputs.
3. **Inference** — Encode a test input grid and decode it to produce a predicted output, then generate a `submission.json` file.

## Model Architecture

| Component | Details |
|---|---|
| Grid size | 30 x 30 (padded), 10 color classes (0–9) |
| Patch size | 5 x 5 |
| Embedding dim | 192 |
| Encoder | 6 Transformer blocks, 6 attention heads |
| Decoder | Transformer blocks with learnable mask tokens |
| VICReg projector | MLP with 2048 hidden/output dims |
| Positional encoding | 2D sinusoidal |
| Default mask ratio | 0.75 |

## Setup

**1. Create a virtual environment (recommended):**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

**2. Install dependencies:**

```bash
pip install -r requirements.txt
```

**3. Prepare data:**

Place the ARC dataset JSON files in the project directory (or specify paths via CLI args):

- `arc-agi_training_challenges.json`
- `arc-agi_training_solutions.json`
- `arc-agi_evaluation_challenges.json` (for eval)
- `arc-agi_test_challenges.json` (for test submission)

## Usage

### Quick Start — Full Workflow

```bash
# 1. Pre-train the encoder
python arc_solver_vicreg.py --pretrain \
    --model_save_path models/encoder.pth \
    --epochs 50 --lr 0.0003 --mask_ratio 0.75 --batch_size 16

# 2. Fine-tune the decoder
python arc_solver_vicreg.py --train_decoder \
    --model_save_path models/encoder.pth \
    --decoder_epochs 30 --decoder_lr 0.0001 --batch_size 16

# 3. Solve the test set
python arc_solver_vicreg.py --solve_test \
    --model_save_path models/encoder.pth \
    --submission_file submissions/submission.json
```

### Command-Line Arguments

**Action flags** (at least one required):

| Flag | Description |
|---|---|
| `--pretrain` | Run VICReg self-supervised pre-training |
| `--train_decoder` | Train the decoder on a pre-trained encoder |
| `--solve_eval` | Solve evaluation set and save predictions |
| `--solve_test` | Solve test set and save predictions |

**Key parameters:**

| Argument | Default | Description |
|---|---|---|
| `--model_save_path` | — | Path to save/load the trained model |
| `--submission_file` | — | Output path for submission JSON |
| `--batch_size` | 16 | Training batch size |
| `--epochs` | 50 | VICReg pre-training epochs |
| `--decoder_epochs` | 30 | Decoder fine-tuning epochs |
| `--lr` | 3e-4 | Pre-training learning rate |
| `--decoder_lr` | 1e-4 | Decoder learning rate |
| `--mask_ratio` | 0.75 | Fraction of patches masked during pre-training |
| `--debug_print_batch_zero` | off | Print detailed logs for the first batch |

**Data paths:**

| Argument | Description |
|---|---|
| `--training_challenges` | Path to training challenges JSON |
| `--training_solutions` | Path to training solutions JSON |
| `--eval_challenges` | Path to evaluation challenges JSON |
| `--test_challenges` | Path to test challenges JSON |

Run `python arc_solver_vicreg.py --help` for all options.

## Code Structure

### Data Pipeline
- `load_arc_data()` — Load JSON dataset files
- `grid_to_tensor()` / `tensor_to_grid()` — Convert between ARC grids and tensors
- `ARCDataset` — PyTorch Dataset with modes: `vicreg_pretrain`, `decoder_train`, `eval`, `test`

### Model (`ARCVicregViT`)
- `PatchEmbed` — Grid-to-patch embedding
- `Attention` — Multi-head self-attention
- `ViTBlock` — Transformer block (used in both encoder and decoder)
- `Projector` — MLP for VICReg projections
- `forward_pretrain()` — VICReg pre-training forward pass
- `forward_train_decoder()` — Supervised decoder training forward pass
- `forward_solve()` — Inference forward pass

### Training
- `train_one_epoch_unified()` — Unified training loop for both VICReg and decoder phases
- `solve_task()` — Run inference on a single ARC task

## Experimental Findings

Initial experiments revealed several challenges with applying ViT + VICReg to ARC:

- **Rapid convergence** — Training epochs complete very quickly, suggesting the model may converge to trivial solutions rather than learning meaningful abstractions.
- **Similarity loss too low** — The VICReg similarity component drops quickly, indicating overly similar representations across masked views (potential representational collapse).
- **StdLoss dominance** — Standard deviation loss dominates the total VICReg loss, suggesting insufficient feature variance across batches.
- **Mask ratio effects** — Higher mask ratios (e.g., 0.75) produce initially more distinct view representations, but don't resolve the fundamental learning dynamics.

These results suggest that standard ViT inductive biases (designed for natural images) may not align well with ARC's abstract, symbolic, relational patterns when applied to raw pixel grids. Future directions could include object-centric architectures, symbolic reasoning modules, or program synthesis approaches.

## Requirements

- Python 3.8+
- PyTorch >= 1.8.0
- NumPy >= 1.19.0
- tqdm >= 4.50.0

## License

This project is for research and educational purposes.
