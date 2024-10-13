import einops
import torch.nn as nn
import torch
from torch.nn.functional import cosine_similarity
from transformers import AutoModel
from peft import LoraConfig, get_peft_model

class FinalLayers(nn.Module):
    def __init__(self, hidden_size, max_seq_length, supplementary_data_dim, max_fragments):
        super(FinalLayers, self).__init__()

        self.max_fragments = max_fragments
        
        self.layer1 = nn.Linear(max_seq_length, 128)
        self.activation1 = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size + supplementary_data_dim, 16)
        self.layer3 = nn.Linear(16 * 128, 2 * self.max_fragments)

    def forward(self, x, supplementary_data):

        # h x 512 -> h x 128
        x_transposed = einops.rearrange(x, 'b s h -> b h s')
        x = self.layer1(x_transposed)
        x = self.activation1(x)

        # Supplementary data is a list of [b, 75] floats, explode it to [b, 75, 128]
        supplementary_data = einops.repeat(supplementary_data, 'b d -> b d s', s=x.shape[2])

        # h x 128 -> (h + 75) x 128
        x = torch.cat([x, supplementary_data], dim=1)
        
        # (h + 75) x 128 -> 16 x 128
        x_transposed = einops.rearrange(x, 'b h s -> b s h')
        x = self.layer2(x_transposed)

        # 16 x 128 -> 2048 x 1 (flattened)
        x = einops.rearrange(x, 'b s h -> b 1 (s h)')

        # 2048 x 1 -> (2 * max_fragments) x 1
        x = self.layer3(x)

        return x

def calculate_loss(predicted_output, labels):
    # return greedy_cosine_similarity_for_interleaved(predicted_output, labels)
    return nn.MSELoss()(predicted_output, labels)

def extract_fragments_from_interleaved(interleaved_vec):
    mzs = interleaved_vec[:, ::2]  # Extract m/z values (even indices across batch)
    intensities = interleaved_vec[:, 1::2]  # Extract intensities (odd indices across batch)
    return mzs, intensities

def sort_two_by_first(arr1, arr2, reverse=True):
    """
    Sorts two tensors based on the values of the first tensor,
    maintaining the corresponding elements of the second tensor in the same order.

    Parameters:
    - arr1: The tensor to sort by (2D tensor, batch x n values).
    - arr2: The tensor whose order corresponds to arr1 (2D tensor, batch x n values).
    - reverse: Boolean indicating whether to sort in descending order (default is True).

    Returns:
    - Two sorted 2D tensors.
    """
    sorted_indices = torch.argsort(arr1, dim=1, descending=reverse)
    arr1_sorted = torch.gather(arr1, 1, sorted_indices)
    arr2_sorted = torch.gather(arr2, 1, sorted_indices)
    return arr1_sorted, arr2_sorted

def greedy_cosine_similarity_for_interleaved(pred_vec, actual_vec):
    pred_mzs, pred_intentensities = extract_fragments_from_interleaved(pred_vec)
    actual_mzs, actual_intensities = extract_fragments_from_interleaved(actual_vec)

    sorted_pred_mzs, sorted_pred_intensities = sort_two_by_first(pred_mzs, pred_intentensities)
    sorted_actual_mzs, sorted_actual_intensities = sort_two_by_first(actual_mzs, actual_intensities)

    # Apply cosine similarity along each batch (dim=1)
    similarity_mzs_1 = torch.abs(cosine_similarity(sorted_pred_mzs.float(), sorted_actual_mzs.float(), dim=1))
    similarity_intensities_1 = torch.abs(cosine_similarity(sorted_pred_intensities.float(), sorted_actual_intensities.float(), dim=1))

    similarity_1 = (similarity_mzs_1 + similarity_intensities_1) / 2

    sorted_pred_intensities, sorted_pred_mzs = sort_two_by_first(pred_intentensities, pred_mzs)
    sorted_actual_intensities, sorted_actual_mzs = sort_two_by_first(actual_intensities, actual_mzs)

    similarity_mzs_2 = torch.abs(cosine_similarity(sorted_pred_mzs.float(), sorted_actual_intensities.float(), dim=1))
    similarity_intensities_2 = torch.abs(cosine_similarity(sorted_pred_intensities.float(), sorted_actual_mzs.float(), dim=1))

    similarity_2 = (similarity_mzs_2 + similarity_intensities_2) / 2

    # Return the maximum similarity for each batch
    return torch.max(similarity_1, similarity_2)



class CustomChemBERTaModel(nn.Module):
    def __init__(self, model, max_fragments, max_seq_length, supplementary_data_dim):
        super(CustomChemBERTaModel, self).__init__()
        self.model = model
        self.max_fragments = max_fragments
        self.max_seq_length = max_seq_length
        
        # Get hidden size from the ChemBERTa model configuration
        self.hidden_size = self.model.config.hidden_size

        # Dimension of the supplementary data
        self.dim_supplementary_data = supplementary_data_dim

        # We'll initialize final_layers in the forward method
        self.final_layers = FinalLayers(self.hidden_size, self.max_seq_length, self.dim_supplementary_data, self.max_fragments)

    def forward(self, input_ids, attention_mask, supplementary_data, labels):
        # Pass inputs through ChemBERTa
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)

        # Extract last hidden state (embeddings)
        last_hidden_state = outputs.hidden_states[-1]  

        # Pad the hidden state to max_seq_length
        # padded_hidden_state = nn.functional.pad(last_hidden_state, (0, 0, 0, self.max_seq_length - last_hidden_state.shape[1]))
        
        # Flatten the padded hidden state
        # flatten_hidden_state = einops.rearrange(padded_hidden_state, 'b s h -> b (s h)')

        # Concatenate with supplementary data
        # state = torch.cat([flatten_hidden_state, supplementary_data], dim=1)

        # Pass through the final layers
        predicted_output = self.final_layers(last_hidden_state, supplementary_data)  # Shape: [batch_size, 2 * max_fragments]

        # Drop singular dimensions
        predicted_output = predicted_output.squeeze()

        # Calculate the loss by comparing to labels
        loss = calculate_loss(predicted_output, labels)

        return loss, predicted_output



if __name__ == "__main__":
    # Example interleaved lists (padded and converted to tensors)

    model = AutoModel.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
    MS_model = CustomChemBERTaModel(model, 256, 512)

    print(MS_model)

    for name, param in MS_model.named_parameters():
        if param.requires_grad:
            print(f"{name} has shape {param.shape}")