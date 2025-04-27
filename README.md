# Fine-Tuning Llama 3.2 Instruct with Unsloth

## Project Overview

This project demonstrates how to fine-tune a Meta Llama 3.2 Instruct model (specifically `unsloth/Llama-3.2-3B-Instruct` or whatever you want) using the Unsloth library for efficient LoRA (Low-Rank Adaptation) training. The goal is to adapt the pre-trained model to a specific conversational style using the `mlabonne/FineTome-100k` dataset.

The process is divided into three main steps, each handled by a separate Python script:
1.  **Downloading Resources:** Fetching the base model, tokenizer, and dataset.
2.  **Fine-Tuning:** Training the LoRA adapters on the downloaded data.
3.  **Inference:** Testing the fine-tuned model with sample prompts from the dataset.

## Key Components

* **Model:** `unsloth/Llama-3.2-3B-Instruct` (or `1B`) - A Llama 3.2 model optimized by Unsloth.
* **Library:** `unsloth` - Used for faster training and reduced memory usage via LoRA and 4-bit quantization.
* **Dataset:** `mlabonne/FineTome-100k` - A dataset containing conversational data in ShareGPT format.
* **Scripts:**
    * `download_model_dataset.py`: Handles downloading the model, tokenizer, and dataset.
    * `finetune.py`: Performs the LoRA fine-tuning process.
    * `test_usage.py`: Runs inference using the fine-tuned model.

## Prerequisites

1.  **Python Environment:** Python 3.9+ is recommended.
2.  **GPU:** A CUDA-enabled GPU is highly recommended for reasonable training times. Unsloth is optimized for NVIDIA GPUs (T4, Ampere, etc.).
3.  **Dependencies:** Install the required Python libraries. You can typically install them using pip:
    ```bash
    # Install Unsloth (includes dependencies like torch, transformers, peft, accelerate, bitsandbytes)
    # Follow official Unsloth installation instructions for your OS/environment:
    # [https://github.com/unslothai/unsloth#installation](https://github.com/unslothai/unsloth#installation)
    ```
    *Note: Refer to the original `llama3_2_(1b_and_3b)_conversational.py` or the Unsloth documentation for the exact dependencies and versions if you encounter issues.*

## How to Run

Execute the scripts sequentially in the following order:

1.  **Download Resources:**
    * Run the script to download the base model, tokenizer, and the FineTome dataset.
    * This will create `./model` and `./dataset` directories.
    ```bash
    python download_model_dataset.py
    ```

2.  **Fine-Tune the Model:**
    * Run the script to load the downloaded resources, apply LoRA adapters, perform fine-tuning, and save the trained adapters.
    * This script reads from `./model` and `./dataset`, and saves the adapters to `./lora_model`. Checkpoints are saved in `./outputs`.
    ```bash
    python finetune.py
    ```
    * *Note:* This step requires a GPU and may take some time depending on your hardware and the `max_steps` configuration in the script.

3.  **Test Inference:**
    * Run the script to load the base model, merge the fine-tuned LoRA adapters from `./lora_model`, and test the resulting model by generating responses for 5 random samples from the dataset.
    ```bash
    python test_usage.py
    ```
    * Observe the output in the console to see the model's generated responses.

Or you can just run:
 ```bash
 sbatch run_finetune.sh
 ```

