from config import *

from transformers import AutoModelForMaskedLM, AutoTokenizer
from src.team5.data_preprocessing import preprocess_data
from model_definition import create_custom_model

from profiling import create_profiling_callback

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

if __name__ == "__main__":
    print("Configuration loaded:")
    print(f"Dataset: {DATASET}")
    print(f"Base Model: {BASE_MODEL}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Number of Epochs: {NUM_EPOCHS}")
    print(f"Profiling Enabled: {ENABLE_PROFILING}")


    # Set up device
    device, use_data_parallel = setup_device()


    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer()
    print("Model and tokenizer loaded successfully.")
    # Add your training logic here

    # Create custom model
    custom_model = create_custom_model(model)
    print("Custom model created successfully.")

    # Create PEFT model
    peft_model = create_peft_model(custom_model)
    peft_model = peft_model.to(device)
    if use_data_parallel:
        peft_model = torch.nn.DataParallel(peft_model)
    print("PEFT model created successfully and moved to device.")

    # Ensure all parameters are on the correct device
    for param in peft_model.parameters():
        param.data = param.data.to(device)

    if ENABLE_PROFILING:
        # Print where each tensor is placed
        for name, param in peft_model.named_parameters():
            if param.requires_grad:
                print(f"{name} is placed on {param.device}")

    
    # Preprocess data
    train_dataset, test_dataset = preprocess_data()
    print("Data preprocessing completed.")
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

     # Preprocess data
    train_dataset, test_dataset = preprocess_data()
    print("Data preprocessing completed.")
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    # Setup Weights & Biases
    wandb_enabled = setup_wandb()

    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=f"../logs/training_{date.today().strftime('%Y-%m-%d')}-{datetime.now().strftime('%H-%M-%S')}",
        num_train_epochs=NUM_EPOCHS,
        dataloader_num_workers=8,
        evaluation_strategy="steps",
        logging_steps=0.01,
        eval_steps=0.05,
        label_names=["labels"],
        report_to="wandb" if wandb_enabled else "none",
        auto_find_batch_size=(device.type == "cuda"),
        use_mps_device=(device.type == "mps"),
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
    )

    # Set up callbacks
    callbacks = []
    if ENABLE_PROFILING:
        callbacks.append(create_profiling_callback(device, train_dataset, peft_model))

    # Create Trainer
    trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        callbacks=callbacks,
    )

    # Start training
    print("Starting training...")
    trainer.train()

    print("Training completed.")