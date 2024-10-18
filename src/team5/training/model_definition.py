from src.team5.models.custom_model import CustomChemBERTaModel

from .config import MAX_FRAGMENTS, MAX_SEQ_LENGTH, SUPPLEMENTARY_DATA_DIM


def create_custom_model(base_model):
    custom_model = CustomChemBERTaModel(
        base_model, MAX_FRAGMENTS, MAX_SEQ_LENGTH, SUPPLEMENTARY_DATA_DIM
    )

    print("Custom Model Structure:")
    print(custom_model)

    print("\nTrainable parameters:")
    for name, param in custom_model.named_parameters():
        if param.requires_grad:
            print(f"{name} has shape {param.shape}")

    return custom_model


if __name__ == "__main__":
    # This is just for testing the model definition independently
    from transformers import AutoModelForMaskedLM

    from config import BASE_MODEL

    base_model = AutoModelForMaskedLM.from_pretrained(BASE_MODEL)
    custom_model = create_custom_model(base_model)
    print("Custom model created successfully.")
