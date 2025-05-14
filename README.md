# ViT with VICReg for Abstraction and Reasoning Corpus (ARC)

This repository contains a Python script for applying a Vision Transformer (ViT) model, pre-trained using VICReg (Variance-Invariance-Covariance Regularization) self-supervised learning, to solve tasks from the Abstraction and Reasoning Corpus (ARC). The project allows for separate VICReg pre-training, supervised decoder fine-tuning, and generation of submission files for ARC tasks.

## Project Structure

- `arc_solver_vicreg.py`: The main Python script containing the model definition, data handling, training loops, and inference logic.
- `requirements.txt`: Lists the necessary Python dependencies.
- `data/` (Create this directory manually): Place ARC dataset JSON files here (e.g., `arc-agi_training_challenges.json`).
- `models/` (Optional, will be created by script): Directory where trained models will be saved.
- `submissions/` (Optional, will be created by script): Directory where output submission files will be saved.

## Features

- Vision Transformer (ViT) architecture adapted for grid-based ARC tasks.
- Self-supervised pre-training of the ViT encoder using the VICReg objective.
  - Includes a CLS token for global representation during pre-training.
  - Random masking of input patches to create different "views".
- Supervised training of a decoder head to predict output ARC grids.
- Inference mode to solve ARC tasks and generate a `submission.json` file.
- Configurable hyperparameters via command-line arguments.
- Detailed debugging options for analyzing the pre-training process.

## Setup

1.  **Clone the repository (or create the files):**
    If you have set up a Git repository for this project, clone it. Otherwise, ensure `arc_solver_vicreg.py` and `requirements.txt` are in your project directory.

2.  **Create a Python virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download ARC Data:**
    Obtain the ARC dataset JSON files (e.g., from the official ARC GitHub repository or Kaggle competition). Create a `data/` directory in your project root (or use the default location, which is the same directory as the script) and place the JSON files there.
    The script defaults to looking for files like:
    - `arc-agi_training_challenges.json`
    - `arc-agi_training_solutions.json`
    - `arc-agi_evaluation_challenges.json`
    - `arc-agi_test_challenges.json`
    You can specify different paths using command-line arguments.

## Usage

The script `arc_solver_vicreg.py` is controlled via command-line arguments.

### Common Arguments:

- `--model_save_path`: Path to save or load the trained model (e.g., `models/arc_model.pth`).
- `--submission_file`: Path to save the output submission JSON (e.g., `submissions/submission.json`).
- `--batch_size`: Batch size for training.
- `--num_workers`: Number of DataLoader workers.
- `--mask_ratio`: Ratio of patches to mask during VICReg pre-training (default: 0.75).
- `--debug_print_batch_zero`: Enable detailed logs for the first batch of the first pre-training epoch.
- `--training_challenges`, `--training_solutions`, `--eval_challenges`, `--test_challenges`: Paths to the respective dataset JSON files.

### Training Phases:

1.  **VICReg Pre-training:**
    This phase trains the ViT encoder and the VICReg projector using self-supervision.
    ```bash
    python arc_solver_vicreg.py \
        --pretrain \
        --training_challenges data/arc-agi_training_challenges.json \
        --training_solutions data/arc-agi_training_solutions.json \
        --model_save_path models/vicreg_pretrained.pth \
        --epochs 50 \
        --lr 0.0003 \
        --mask_ratio 0.75 \
        --batch_size 16 \
        --debug_print_batch_zero 
        # Add other relevant hyperparameters as needed
    ```

2.  **Decoder Fine-tuning:**
    This phase trains the decoder head. If a model exists at `--model_save_path` (e.g., from pre-training), its weights (primarily the encoder) will be loaded and frozen by default, and only the decoder will be trained.
    ```bash
    python arc_solver_vicreg.py \
        --train_decoder \
        --training_challenges data/arc-agi_training_challenges.json \
        --training_solutions data/arc-agi_training_solutions.json \
        --model_save_path models/vicreg_finetuned.pth \
        --decoder_epochs 30 \
        --decoder_lr 0.0001 \
        --batch_size 16
        # If models/vicreg_finetuned.pth exists from a previous run and you want to continue,
        # or if it points to the output of the pretrain step (e.g., models/vicreg_pretrained.pth)
    ```

### Solving (Inference):

This phase uses a trained model (encoder + decoder) to predict outputs for ARC tasks.

-   **Solve Evaluation Set:**
    ```bash
    python arc_solver_vicreg.py \
        --solve_eval \
        --eval_challenges data/arc-agi_evaluation_challenges.json \
        --model_save_path models/vicreg_finetuned.pth \
        --submission_file submissions/eval_submission.json
    ```

-   **Solve Test Set:**
    ```bash
    python arc_solver_vicreg.py \
        --solve_test \
        --test_challenges data/arc-agi_test_challenges.json \
        --model_save_path models/vicreg_finetuned.pth \
        --submission_file submissions/test_submission.json
    ```

**Default Behavior:** If no action flags (`--pretrain`, `--train_decoder`, `--solve_eval`, `--solve_test`) are provided, the script will attempt to solve the test set by default, loading a model from the default model save path if it exists.

### Full Example Workflow (Pre-train, then Fine-tune Decoder, then Solve Test):

1.  **Pre-train the encoder:**
    ```bash
    python arc_solver_vicreg.py --pretrain --model_save_path models/encoder_pretrained.pth --epochs 50 --mask_ratio 0.75 --debug_print_batch_zero
    ```
2.  **Fine-tune the decoder (loading the pre-trained encoder):**
    The `--model_save_path` for this step should initially point to the output of the pre-training step if you want to load those weights. The script will then save the model *with the trained decoder* to this same path, overwriting the encoder-only model or updating it.
    ```bash
    python arc_solver_vicreg.py --train_decoder --model_save_path models/encoder_pretrained.pth --decoder_epochs 30 
    # After this, models/encoder_pretrained.pth will contain the encoder (frozen) + trained decoder.
    # You might want to use a different save path for the fully fine-tuned model, e.g.:
    # python arc_solver_vicreg.py --train_decoder --model_save_path models/full_model_finetuned.pth --decoder_epochs 30
    # (This assumes models/encoder_pretrained.pth was loaded if it existed and --pretrain was not set in this command)
    ```
3.  **Solve the Test Set:**
    Load the model that has both the (potentially pre-trained) encoder and the trained decoder.
    ```bash
    python arc_solver_vicreg.py --solve_test --model_save_path models/full_model_finetuned.pth --submission_file submissions/my_final_submission.json
    ```

For a full list of command-line options and their default values:
```bash
python arc_solver_vicreg.py --help
Code Structure OverviewConfiguration Constants: Global defaults for paths, model, and training parameters defined at the top of arc_solver_vicreg.py.Data Loading & Preprocessing:load_arc_data(): Loads JSON data.grid_to_tensor(): Converts ARC grids to padded tensors.tensor_to_grid(): Converts tensors back to ARC grids.Positional Embedding Utilities: Functions to generate 2D sinusoidal positional embeddings.Vision Transformer Components:PatchEmbed: Converts image grids into sequences of patch embeddings.Attention: Multi-head self-attention mechanism.Mlp: Feed-forward network for Transformer blocks.ViTBlock: A single Transformer encoder/decoder block.Projector: MLP used by VICReg.ARCVicregViT (Main Model Class):Initializes the encoder, VICReg projector, and decoder components.initialize_weights(): Sets up initial weights and positional embeddings._encode_view_for_vicreg(): Encodes an input grid into two different "views" by applying random masking, for VICReg pre-training.forward_vicreg_loss(): Computes the three components of the VICReg loss (similarity, variance, covariance).forward_pretrain(): Defines the forward pass for the VICReg pre-training stage._encode_input_grid(): Encodes a single input grid for use by the decoder during fine-tuning or inference._decode_to_output_grid_logits(): Takes encoded input features and generates output grid predictions using the decoder.forward_train_decoder(): Defines the forward pass for the supervised decoder training stage.forward_solve(): Defines the forward pass for inference (solving tasks).set_*_trainable(): Helper methods to toggle the requires_grad status for different parts of the model (encoder, projector, decoder).ARCDataset: PyTorch Dataset class for loading and transforming ARC tasks for different modes (pre-training, decoder training, evaluation/test).Training and Evaluation Functions:train_one_epoch_unified(): A unified training loop function that handles one epoch of either VICReg pre-training or decoder training, based on a configuration dictionary.solve_task(): Processes a single ARC task input using the trained model to produce a predicted output grid.Main Execution Logic (if __name__ == "__main__":)Parses command-line arguments.Initializes the model, dataset, and optimizer based on the specified action.Orchestrates the selected operational phases (pre-train, train_decoder, solve).Experimental Findings (Summary from Development)During the development and testing of this codebase, initial experiments applying this ViT+VICReg architecture to the ARC dataset highlighted significant challenges in achieving effective learning. Key observations included:Rapid Training Epochs: Both pre-training and decoder training epochs completed very quickly, suggesting that the model was not engaging in computationally deep learning or was converging to trivial solutions.VICReg Loss Behavior:The similarity loss (SimLoss) component of VICReg, which aims to make representations of different views of the same input similar, often started low or decreased rapidly to a low value. This indicated that the encoder was producing overly similar global representations (CLS token embeddings) for the two masked views, even when the masks themselves were different.The standard deviation loss (StdLoss) frequently dominated the total VICReg loss, suggesting the model struggled to maintain sufficient variance in its learned features across the batch, a potential sign of representational collapse.Impact of Mask Ratio: Increasing the MASK_RATIO for VICReg (e.g., from 0.5 to 0.75) did lead to initially more distinct representations from the encoder for the two views. However, the overall learning dynamics and rapid training times persisted.These observations suggest a fundamental difficulty for this standard Vision Transformer architecture, even when augmented with VICReg self-supervision, to learn the abstract, symbolic, and relational patterns inherent in ARC tasks when applied directly to raw pixel grids. The inductive biases of ViTs, primarily developed for natural images, may not align well with
