import einops
import torch.nn as nn
import torch
from torch.nn.functional import cosine_similarity
from transformers import AutoModel
from peft import LoraConfig, get_peft_model

class FinalLayers(nn.Module):
    def __init__(self, input_size, max_fragments):
        super(FinalLayers, self).__init__()
        
        intermediate_size = (input_size + 2 * max_fragments) // 2  # Average of input and output sizes
        
        self.layer1 = nn.Linear(input_size, intermediate_size)
        self.activation = nn.GeLU()
        self.layer2 = nn.Linear(intermediate_size, 2 * max_fragments)
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.activation(x)
        x = self.layer2(x)
        return x

def calculate_loss(predicted_output, labels):
    return greedy_cosine_similarity_for_interleaved(predicted_output, labels)

def extract_fragments_from_interleaved(interleaved_vec):
    """
    Splits the interleaved vector into m/z values and intensities.
    Even indices contain m/z values and odd indices contain intensities.
    
    Parameters:
    - interleaved_vec: A tensor where even indices are m/z and odd indices are intensities.

    Returns:
    - Two tensors: m/z values and intensities.
    """
    mzs = interleaved_vec[::2]  # Extract m/z values (even indices)
    intensities = interleaved_vec[1::2]  # Extract intensities (odd indices)
    return mzs, intensities

def sort_two_by_first(arr1, arr2, reverse=True):
    """
    Sorts two arrays (lists or PyTorch 1D tensors) based on the values of the first array,
    maintaining the corresponding elements of the second array in the same order.

    Parameters:
    - arr1: The array to sort by (list or 1D tensor).
    - arr2: The array whose order corresponds to arr1 (list or 1D tensor).
    - reverse: Boolean indicating whether to sort in descending order (default is True).

    Returns:
    - Two sorted 1D tensors.
    """
    sorted_indices = torch.argsort(arr1, descending=reverse)
    arr1_sorted = arr1[sorted_indices]
    arr2_sorted = arr2[sorted_indices]
    return arr1_sorted, arr2_sorted

def greedy_cosine_similarity_for_interleaved(pred_vec, actual_vec):
    """
    Computes cosine similarity between two interleaved vectors by applying two sorting strategies:
    1. Sorting based on m/z values.
    2. Sorting based on intensity values.
    
    It then returns the maximum cosine similarity score from both strategies.

    Parameters:
    - pred_vec: Predicted interleaved vector (m/z and intensities).
    - actual_vec: Actual interleaved vector (m/z and intensities).

    Returns:
    - Maximum cosine similarity score between the two sorting strategies.
    """
    # Extract m/z and intensity values from interleaved vectors
    pred_mzs, pred_intensities = extract_fragments_from_interleaved(pred_vec)
    actual_mzs, actual_intensities = extract_fragments_from_interleaved(actual_vec)

    # Sorting by m/z values first
    sorted_pred_mzs, sorted_pred_intensities = sort_two_by_first(pred_mzs, pred_intensities)
    sorted_actual_mzs, sorted_actual_intensities = sort_two_by_first(actual_mzs, actual_intensities)

    # Compute cosine similarity for m/z and intensities separately after sorting by m/z
    similarity_mzs_1 = abs(cosine_similarity(sorted_pred_mzs.float(), sorted_actual_mzs.float(), dim=0).item())
    similarity_intensities_1 = abs(cosine_similarity(sorted_pred_intensities.float(), sorted_actual_intensities.float(), dim=0).item())
    similarity_1 = (similarity_mzs_1 + similarity_intensities_1) / 2

    # Sorting by intensity values (swapping m/z and intensity roles)
    sorted_pred_intensities, sorted_pred_mzs = sort_two_by_first(pred_intensities, pred_mzs)
    sorted_actual_intensities, sorted_actual_mzs = sort_two_by_first(actual_intensities, actual_mzs)

    # Compute cosine similarity after sorting by intensities
    similarity_mzs_2 = abs(cosine_similarity(sorted_pred_mzs.float(), sorted_actual_mzs.float(), dim=0).item())
    similarity_intensities_2 = abs(cosine_similarity(sorted_pred_intensities.float(), sorted_actual_intensities.float(), dim=0).item())
    similarity_2 = (similarity_mzs_2 + similarity_intensities_2) / 2

    # Return the maximum similarity score from the two sorting strategies
    return max(similarity_1, similarity_2)


class CustomChemBERTaModel(nn.Module):
    def __init__(self, model, max_fragments, max_seq_length):
        super(CustomChemBERTaModel, self).__init__()
        self.model = model
        self.max_fragments = max_fragments
        self.max_seq_length = max_seq_length
        
        # Get hidden size from the ChemBERTa model configuration
        self.hidden_size = self.model.config.hidden_size
        
        # We'll initialize final_layers in the forward method
        self.final_layers = None

    def forward(self, input_ids, supplementary_data, labels):
        # Pass inputs through ChemBERTa
        outputs = self.model(input_ids=input_ids, output_hidden_states=True)

        # Extract last hidden state (embeddings)
        last_hidden_state = outputs.hidden_states[-1]  

        # Pad the hidden state to max_seq_length
        padded_hidden_state = nn.functional.pad(last_hidden_state, (0, 0, 0, self.max_seq_length - last_hidden_state.shape[1]))
        
        # Flatten the padded hidden state
        flatten_hidden_state = einops.rearrange(padded_hidden_state, 'b s h -> b (s h)')

        # Concatenate with supplementary data
        state = torch.cat([flatten_hidden_state, supplementary_data], dim=1)

        # Initialize final_layers if not done yet
        if self.final_layers is None:
            input_size = state.shape[1]
            self.final_layers = FinalLayers(input_size, self.max_fragments)

        # Pass through the final layers
        predicted_output = self.final_layers(state)  # Shape: [batch_size, 2 * max_fragments]

        # Calculate the loss by comparing to labels
        loss = calculate_loss(predicted_output, labels)

        return predicted_output, loss


from peft import LoraConfig, get_peft_model

peft_config = LoraConfig(
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["key", "query", "value"] # they seem to drop off the "key" often?
    modules_to_save=["poolers"] # change this to the name of the new modules at the end.
    bias="none"
)



if __name__ == "__main__":
    # Example interleaved lists (padded and converted to tensors)

    model = AutoModel.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
    MS_model = CustomChemBERTaModel(model, 10, 100)
    peft_model = get_peft_model(MS_model, peft_config)
    peft_model.print_trainable_parameters() #check that it's training the right things


    print(MS_model)

    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name} has shape {param.shape}")