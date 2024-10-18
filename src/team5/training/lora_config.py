from peft import LoraConfig, get_peft_model

def create_peft_model(base_model):
    peft_config = LoraConfig(
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["query", "value"],
        modules_to_save=[
            "final_layers"
        ],  # change this to the name of the new modules at the end.
        bias="none",
    )
    
    peft_model = get_peft_model(base_model, peft_config)
    
    print("PEFT model created. Trainable parameters:")
    peft_model.print_trainable_parameters()
    
    print("\nTrainable layers:")
    for name, param in peft_model.named_parameters():
        if param.requires_grad:
            print(f"{name} is trainable")
    
    return peft_model

if __name__ == "__main__":
    # This is just for testing the LoRA configuration independently
    from model_definition import create_custom_model
    from transformers import AutoModelForMaskedLM
    from config import BASE_MODEL
    
    base_model = AutoModelForMaskedLM.from_pretrained(BASE_MODEL)
    custom_model = create_custom_model(base_model)
    peft_model = create_peft_model(custom_model)
    print("PEFT model created successfully.")