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
        self.layer2 = nn.Linear(hidden_size + supplementary_data_dim, 3)
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

        # (b, max_fragments, h+75) -> (b, max_fragments, 3)
        x = self.layer2(x)
        x = self.dropout2(x)
        x = self.activation2(x)

                # Apply sigmoid to the third number (flags) to get values near 0 or 1
        mzs = x[:, :, 0]  # First number (labels)
        probs = x[:, :, 1]   # Second number (probs)
        flags = x[:, :, 2]   # Third number (flags)

        flags = torch.sigmoid(flags)  # Apply sigmoid to flags

        # Multiply the first number (labels) by the flag
        mzs = mzs * flags

        # Adjust the second number (probs) to be very negative where the flag is 0
        probs = probs + torch.log(flags + 1e-6)  # Add a small value to avoid log(0)

        # Apply softmax to the adjusted probabilities along the fragment dimension
        probs = torch.softmax(probs, dim=1)

        return mzs, probs, flags

def extract_fragments_from_interleaved(interleaved_vec):
    mzs = interleaved_vec[:, ::2]  # Extract m/z values (even indices across batch)
    intensities = interleaved_vec[:, 1::2]  # Extract intensities (odd indices across batch)
    return mzs, intensities

def extract_predictions(interleaved_vec):
    mzs = interleaved_vec[:, ::3]  # Extract m/z values (even indices across batch)
    intensities = interleaved_vec[:, 1::3]  # Extract intensities (odd indices across batch)
    flag = interleaved_vec[:, 2::3]  # Extract flag (odd indices across batch)
    return mzs, intensities, flag

def calculate_loss(predictions, actual_vec):
    pred_mzs, pred_probabilities, flag = predictions
    actual_mzs, actual_intensities = extract_fragments_from_interleaved(actual_vec)

    # normalise actual_intensities by dividing by the sum along dim 1
    actual_probabilities = actual_intensities / torch.sum(actual_intensities, dim=1, keepdim=True)

    return gaussian_cross_entropy_loss(actual_mzs, actual_probabilities, pred_mzs, pred_probabilities)


def gaussian_cross_entropy_loss(actual_mzs, actual_probabilities, predicted_mzs, predicted_probabilities, sigma=1.0):
    """
    Computes the cross-entropy loss between the actual and predicted distributions using
    Gaussian kernel density estimation with the log-sum-exp trick for numerical stability.

    Parameters:
    - actual_mzs: Tensor of shape (B, N_actual)
    - actual_probabilities: Tensor of shape (B, N_actual)
    - predicted_mzs: Tensor of shape (B, N_pred)
    - predicted_probabilities: Tensor of shape (B, N_pred)
    - sigma: Standard deviation of the Gaussian kernel (default: 1.0)

    Returns:
    - loss: Scalar tensor representing the cross-entropy loss
    """
    # Ensure the predicted probabilities are not zero to avoid log of zero
    epsilon = 1e-10
    predicted_probabilities = predicted_probabilities.clamp(min=epsilon)


    # Compute the squared differences divided by sigma
    D = ((actual_mzs.unsqueeze(2) - predicted_mzs.unsqueeze(1)) ** 2) / sigma

    # Compute the log of predicted probabilities and reshape for broadcasting
    log_predicted_probs = torch.log(predicted_probabilities).unsqueeze(1)

    # Compute the terms inside the log-sum-exp
    terms = log_predicted_probs - D

    # Apply the log-sum-exp trick along the predicted mzs axis
    log_p_predicted = torch.logsumexp(terms, dim=2)

    # Compute the cross-entropy loss
    loss = -torch.sum(actual_probabilities * log_p_predicted, dim=1)

    loss = loss.mean()

    return loss


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

        predicted_output = self.final_layers(last_hidden_state, supplementary_data, attention_mask)  # Shape: [batch_size, max_fragments, 3]

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
