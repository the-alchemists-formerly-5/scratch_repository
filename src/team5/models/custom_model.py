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
        
        self.layer1 = nn.Linear(max_seq_length, 512)
        self.activation1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.1)
        self.layernorm1 = nn.LayerNorm(512)
        self.layer2 = nn.Linear(hidden_size + supplementary_data_dim, 2)
        self.activation2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.1)
        self.layernorm2 = nn.LayerNorm(2)


    def forward(self, x, supplementary_data, attention_mask):
        # (b, s, h) -> (b, h, s)
        x = einops.rearrange(x, 'b s h -> b h s')

        # Expand the attention mask to match the shape of x and ensure it is a float tensor
        expanded_mask = attention_mask.unsqueeze(1).expand_as(x).float()

        # Apply the attention mask to zero out padding tokens
        x = x * expanded_mask

        # Supplementary data is a list of [b, 75] floats, explode it to [b, 75, 128]
        supplementary_data = einops.repeat(supplementary_data, 'b d -> b d s', s=x.shape[2])

        # (b, h, s) -> (b, h + 75, s)
        x = torch.cat([x, supplementary_data], dim=1)

        # (b, h + 75, s) -> (b, h + 75, max_fragments)
        x = self.layer1(x)
        x = self.dropout1(x)
        x = self.layernorm1(x)
        x = self.activation1(x)

        # (b, h+75, max_fragments) -> (b, max_fragments, h+75)
        x = einops.rearrange(x, 'b h s -> b s h')

        # (b, max_fragments, h+75) -> (b, max_fragments, 2)
        x = self.layer2(x)
        x = self.dropout2(x)
        x = self.layernorm2(x)
        x = self.activation2(x)

        # (b, max_fragments, 2) -> (b, 1, max_fragments * 2)
        x = einops.rearrange(x, 'b s h -> b 1 (s h)')

        return x

def calculate_loss(pred_vec, actual_vec):
    pred_mzs, pred_intensities = extract_fragments_from_interleaved(pred_vec)
    actual_mzs, actual_intensities = extract_fragments_from_interleaved(actual_vec)

    # Convert to tensors and ensure gradients are tracked
    pred_mzs = torch.tensor(pred_mzs, requires_grad=True)
    pred_intensities = torch.tensor(pred_intensities, requires_grad=True)
    actual_mzs = torch.tensor(actual_mzs)
    actual_intensities = torch.tensor(actual_intensities)

    # Remove zero intensities to avoid division by zero
    pred_mask = pred_intensities > 0
    actual_mask = actual_intensities > 0

    pred_mzs = pred_mzs[pred_mask]
    pred_intensities = pred_intensities[pred_mask]
    actual_mzs = actual_mzs[actual_mask]
    actual_intensities = actual_intensities[actual_mask]

    # Compute the Sinkhorn loss
    loss_value = sinkhorn_loss(pred_mzs, pred_intensities, actual_mzs, actual_intensities)

    return loss_value



def extract_fragments_from_interleaved(interleaved_vec):
    mzs = interleaved_vec[:, ::2]  # Extract m/z values (even indices across batch)
    intensities = interleaved_vec[:, 1::2]  # Extract intensities (odd indices across batch)
    return mzs, intensities

def sinkhorn_loss(pred_mzs, pred_intensities, actual_mzs, actual_intensities, epsilon=0.01, num_iters=100):
    # Normalize intensities to sum to 1 (probability distributions)
    pred_intensities = pred_intensities / torch.sum(pred_intensities)
    actual_intensities = actual_intensities / torch.sum(actual_intensities)

    # Compute the cost matrix (absolute differences in m/z)
    C = torch.abs(pred_mzs.unsqueeze(1) - actual_mzs.unsqueeze(0))  # Shape: (N_pred, N_actual)

    # Initialize the kernel matrix
    K = torch.exp(-C / epsilon)

    # Initialize dual variables
    u = torch.ones_like(pred_intensities)
    v = torch.ones_like(actual_intensities)

    # Small epsilon to prevent division by zero
    tiny_value = 1e-6

    # Sinkhorn iterations
    for _ in range(num_iters):
        u = pred_intensities / (K @ v + tiny_value)
        v = actual_intensities / (K.T @ u + tiny_value)

    # Compute the transport plan
    P = torch.diag(u) @ K @ torch.diag(v)

    # Compute the Sinkhorn distance
    loss_value = torch.sum(P * C)

    return loss_value


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
        predicted_output = self.final_layers(last_hidden_state, supplementary_data, )  # Shape: [batch_size, 2 * max_fragments]

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