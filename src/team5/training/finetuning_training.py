from datetime import datetime
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer, get_scheduler
from torch.optim import AdamW
from torch.utils.data import DataLoader
from accelerate import Accelerator
from tqdm.auto import tqdm

from src.team5.training.config import *
from src.team5.training.data_preprocessing import preprocess_data
from src.team5.training.lora_config import create_peft_model
from src.team5.training.model_definition import create_custom_model


def load_model_and_tokenizer():
    model = AutoModelForMaskedLM.from_pretrained(BASE_MODEL)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    return model, tokenizer


# Your main training code will go here


def setup_device():
    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        print(f"Using DataParallel with {num_gpus} GPUs")
        device_type = "cuda"
        use_data_parallel = True
    elif torch.cuda.is_available():
        print("Using a single GPU")
        device_type = "cuda"
        use_data_parallel = False
    elif torch.backends.mps.is_available():
        print("Using MPS (Metal Performance Shaders)")
        device_type = "mps"
        use_data_parallel = False
    else:
        print("Using CPU")
        device_type = "cpu"
        use_data_parallel = False

    device = torch.device(device_type)
    return device, use_data_parallel


def setup_wandb():
    wandb_enabled = os.getenv("WANDB_API_KEY") is not None
    if wandb_enabled:
        os.environ["WANDB_PROJECT"] = "hackathon"
        os.environ["WANDB_LOG_MODEL"] = "end"
        os.environ["WANDB_WATCH"] = "false"
    return wandb_enabled


def main():
    os.environ["WANDB_API_KEY"] = "69f075ac6ff5b82fb8e32313942465d0a23c6ead"

    # Setup Weights & Biases
    wandb_enabled = setup_wandb()

    accelerator = Accelerator(log_with="wandb" if wandb_enabled else None)

    if not wandb_enabled:
        accelerator.print("WANDB_API_KEY not set. Skipping Weights & Biases.")

    accelerator.print("Configuration loaded:")
    accelerator.print(f"Dataset: {DATASET}")
    accelerator.print(f"Base Model: {BASE_MODEL}")
    accelerator.print(f"Batch Size: {BATCH_SIZE}")
    accelerator.print(f"Number of Epochs: {NUM_EPOCHS}")

    accelerator.init_trackers(
        project_name="hackathon",
        config={"lr": LEARNING_RATE, "batch_size": BATCH_SIZE, "num_epochs": NUM_EPOCHS},
        init_kwargs={"wandb": {}}
    )

    # Set up device
    device = accelerator.device

    gradient_accumulation_steps = 1

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer()
    accelerator.print("Model and tokenizer loaded successfully.")
    # Add your training logic here

    # Create custom model
    custom_model = create_custom_model(model)
    accelerator.print("Custom model created successfully.")

    # Create PEFT model
    peft_model = create_peft_model(custom_model)
    peft_model = peft_model.to(device)
    accelerator.print("PEFT model created successfully and moved to device.")

    # Ensure all parameters are on the correct device
    for param in peft_model.parameters():
        param.data = param.data.to(device)

    if ENABLE_PROFILING:
        # Print where each tensor is placed
        for name, param in peft_model.named_parameters():
            if param.requires_grad:
                accelerator.print(f"{name} is placed on {param.device}")

    # Preprocess data
    train_dataset, test_dataset, _ = preprocess_data()
    accelerator.print("Data preprocessing completed.")
    accelerator.print(f"Train dataset size: {len(train_dataset)}")
    accelerator.print(f"Test dataset size: {len(test_dataset)}")

    optimizer = AdamW(peft_model.parameters(), lr=LEARNING_RATE)

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    eval_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Set up training arguments
    train_dataloader, eval_dataloader, peft_model, optimizer = accelerator.prepare(
        train_dataloader, eval_dataloader, peft_model, optimizer
    )

    num_training_steps = NUM_EPOCHS * len(train_dataloader)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )

    output_dir = data_dir.parent.parent / "checkpoints" / f"model_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    os.makedirs(output_dir, exist_ok=True)

    # Start training
    accelerator.print("Starting training...")
    accelerator.print(f"Output directory: {output_dir}")
    peft_model.train()
    global_step = 0
    grad_norm = None
    for epoch in range(NUM_EPOCHS):
        for step, batch in enumerate(train_dataloader):
            loss, outputs = peft_model(**batch)
            loss = loss / gradient_accumulation_steps
            accelerator.backward(loss)
            if step % gradient_accumulation_steps == 0:
                if accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(peft_model.parameters(), max_norm=7.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            global_step += 1
            if step % 50 == 0:
                accelerator.log({
                    "train_loss": loss.detach().float(),
                    "train_grad_norm": grad_norm.detach().float() if grad_norm is not None else None
                }, step=global_step)
                accelerator.print(f"step {global_step:07d}/{num_training_steps:07d}, loss: {loss.detach().float():.6f}")

        step_dir = f"step_{global_step:07d}"
        saving_dir = os.path.join(output_dir, step_dir)
        accelerator.save_state(saving_dir)

        peft_model.eval()
        accelerator.print(f"Starting evaluation for epoch {epoch}...")
        total_eval_loss = 0
        num_eval_steps = 0
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                loss, outputs = peft_model(**batch)
            total_eval_loss += loss.detach().float()
            num_eval_steps += 1
        mean_eval_loss = total_eval_loss / num_eval_steps
        accelerator.log({
            "eval_loss": mean_eval_loss,
        }, step=global_step)
        accelerator.print(f"\n\nepoch {epoch}, eval loss: {mean_eval_loss:.6f}\n\n")

    accelerator.end_training()

    accelerator.print("Training completed.")


# First run: export NCCL_P2P_DISABLE=1
# then: export PYTHONPATH=$PYTHONPATH:`pwd` (from scratch_repository)
# then run: accelerate check, then run using: accelerate launch src/team5/training/finetuning_training.py
if __name__ == "__main__":
    main()
