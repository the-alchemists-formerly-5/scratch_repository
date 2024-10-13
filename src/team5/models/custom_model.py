import einops
import torch.nn as nn
import torch
from torch.nn.functional import cosine_similarity
from transformers import AutoModel
from peft import LoraConfig, get_peft_model
import torch.nn.functional as F

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
        x = self.activation2(x)

        # (b, max_fragments, 2) -> (b, 1, max_fragments * 2)
        x = einops.rearrange(x, 'b s h -> b 1 (s h)')

        return x

def calculate_loss(pred_vec, actual_vec):
    
    pred_mzs, pred_intensities = extract_fragments_from_interleaved(pred_vec)
    actual_mzs, actual_intensities = extract_fragments_from_interleaved(actual_vec)
    

    # Compute the KL divergence approximation
    loss_value = kl_divergence_approximation(pred_mzs, pred_intensities, actual_mzs, actual_intensities)

    return loss_value


def kl_divergence_approximation(pred_mzs, pred_intensities, actual_mzs, actual_intensities, sigma=2.0, min_mz=0., max_mz=5000., num_points=5000, epsilon=1e-8):
    # Create a range of points to evaluate the distributions
    x = torch.linspace(min_mz, max_mz, num_points).to(pred_mzs.device)

    # print()
    # print('pred_mzs', torch.min(pred_mzs), torch.max(pred_mzs), torch.mean(pred_mzs))
    # print('pred_intensities', torch.min(pred_intensities), torch.max(pred_intensities), torch.mean(pred_intensities))
    # print('actual_mzs', torch.min(actual_mzs), torch.max(actual_mzs), torch.mean(actual_mzs))
    # print('actual_intensities', torch.min(actual_intensities), torch.max(actual_intensities), torch.mean(actual_intensities))
    # print()

    # 2. Create continuous probability distributions
    def create_distribution(mzs, probs):
        # Expand dimensions for broadcasting
        x_expanded = x.unsqueeze(0).unsqueeze(1)  # [1, 1, num_points]
        mzs_expanded = mzs.unsqueeze(2)  # [batch_size, num_fragments, 1]
        probs_expanded = probs.unsqueeze(2)  # [batch_size, num_fragments, 1]
        
        # Calculate Gaussian contributions
        gauss = torch.exp(-0.5 * ((x_expanded - mzs_expanded) / sigma)**2)
        dist = torch.sum(gauss * probs_expanded, dim=1)  # Sum over fragments

        # Normalize the distribution
        dist = dist / torch.sum(dist, dim=1, keepdim=True)
        return dist

    pred_dist = create_distribution(pred_mzs, pred_intensities) + epsilon
    actual_dist = create_distribution(actual_mzs, actual_intensities) + epsilon

    # print('pred_dist', torch.min(pred_dist), torch.max(pred_dist), torch.mean(pred_dist))
    # print('pred_dist.log()', torch.min(pred_dist.log()), torch.max(pred_dist.log()), torch.mean(pred_dist.log()))
    # print('actual_dist', torch.min(actual_dist), torch.max(actual_dist), torch.mean(actual_dist))

    # 3. Calculate KL divergence
    kl_div = F.kl_div(pred_dist.log(), actual_dist, reduction='batchmean', log_target=False)

    # print('kl_div', kl_div)
    # print()
    # print()
    # print(kl_div)

    return kl_div


def extract_fragments_from_interleaved(interleaved_vec):
    mzs = interleaved_vec[:, ::2]  # Extract m/z values (even indices across batch)
    intensities = interleaved_vec[:, 1::2]  # Extract intensities (odd indices across batch)
    return mzs, intensities


def sinkhorn_loss(pred_mzs, pred_intensities, actual_mzs, actual_intensities, epsilon=100, num_iters=1):
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
        predicted_output = self.final_layers(last_hidden_state, supplementary_data, attention_mask)  # Shape: [batch_size, 2 * max_fragments]

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
