from safetensors.torch import load_file
from transformers import AutoModelForMaskedLM

from src.team5.models.custom_model import CustomChemBERTaModel
from src.team5.training.config import (BASE_MODEL, MAX_FRAGMENTS,
                                       MAX_SEQ_LENGTH, SUPPLEMENTARY_DATA_DIM)
from src.team5.training.lora_config import create_peft_model

from .checkpoint_loader import CHECKPOINT_DIR, get_latest_checkpoint


# Custom loading function to handle PEFT layers
def load_peft_state_dict(model, state_dict):
    model_state_dict = model.state_dict()
    for name, param in state_dict.items():
        if name in model_state_dict:
            if param.shape == model_state_dict[name].shape:
                model_state_dict[name].copy_(param)
            else:
                print(f"Skipping parameter {name} due to shape mismatch")
        else:
            print(f"Ignoring unexpected key in state_dict: {name}")

    model.load_state_dict(model_state_dict, strict=False)


def load_model(version=None):
    if version is None:
        checkpoint_dir = get_latest_checkpoint()
    else:
        checkpoint_dir = CHECKPOINT_DIR / version

    model_path = checkpoint_dir / "model.safetensors"

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found in {checkpoint_dir}")

    # Load the base model
    base_model = AutoModelForMaskedLM.from_pretrained(BASE_MODEL)

    # Create the custom model
    custom_model = CustomChemBERTaModel(
        model=base_model,
        max_fragments=MAX_FRAGMENTS,
        max_seq_length=MAX_SEQ_LENGTH,
        supplementary_data_dim=SUPPLEMENTARY_DATA_DIM,
    )

    # Create the PEFT model
    peft_model = create_peft_model(custom_model)

    # Load the trained weights
    state_dict = load_file(str(model_path))

    # Load the state dict using the custom function
    load_peft_state_dict(peft_model, state_dict)
    peft_model.eval()
    return peft_model
