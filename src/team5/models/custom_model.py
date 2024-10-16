import torch.nn as nn
import torch
from torch.nn.functional import cosine_similarity
from transformers import AutoModel
from peft import LoraConfig, get_peft_model
import torch.nn.functional as F
from matchms import Spectrum
from matchms.similarity import CosineGreedy, CosineHungarian

class FinalLayers(nn.Module):
    def __init__(self, hidden_size, max_seq_length, supplementary_data_dim, max_fragments, num_heads):
        super(FinalLayers, self).__init__()

        self.max_fragments = max_fragments

        # Linear layer to process supplementary data
        self.layer1 = nn.Linear(supplementary_data_dim, hidden_size)
        self.activation1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.1)
        self.layernorm1 = nn.LayerNorm(hidden_size)

        # Multihead cross-attention layer
        self.cross_attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, batch_first=True)

        # Multihead self-attention layer
        self.self_attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, batch_first=True)

        # Output linear layer to project to (b, max_fragments, 3)
        self.output_linear = nn.Linear(hidden_size, 3)
        self.dropout2 = nn.Dropout(0.1)
        self.activation2 = nn.ReLU()

    def forward(self, x, supplementary_data, attention_mask):
        # x: (batch_size, seq_length, hidden_size)
        # supplementary_data: (batch_size, supplementary_data_dim)
        # attention_mask: (batch_size, seq_length)

        # Apply the attention mask to zero out padding tokens in x
        expanded_mask = attention_mask.unsqueeze(-1)  # Shape: (batch_size, seq_length, 1)
        x = x * expanded_mask  # Zero out padding tokens

        # Process supplementary data to match hidden_size
        supplementary_data = self.layer1(supplementary_data)  # Shape: (batch_size, hidden_size)
        supplementary_data = self.dropout1(supplementary_data)
        supplementary_data = self.layernorm1(supplementary_data)
        supplementary_data = self.activation1(supplementary_data)

        # Expand supplementary_data to match x's shape and add to x
        supplementary_data_expanded = supplementary_data.unsqueeze(1).expand(-1, x.size(1), -1)  # Shape: (batch_size, seq_length, hidden_size)
        x = x + supplementary_data_expanded

        # Initialize y randomly with shape (batch_size, max_fragments, hidden_size)
        batch_size, _, hidden_size = x.size()
        y = torch.randn(batch_size, self.max_fragments, hidden_size, device=x.device, dtype=x.dtype)

        # Create key_padding_mask for x (True where padding)
        key_padding_mask = attention_mask == 0  # Shape: (batch_size, seq_length)

        # Multihead cross-attention: query from y, key and value from x
        y, _ = self.cross_attention(query=y, key=x, value=x, key_padding_mask=key_padding_mask)
        # y is updated after cross-attention

        # Multihead self-attention on y
        y, _ = self.self_attention(query=y, key=y, value=y)
        # y is further updated after self-attention

        # Project y to (batch_size, max_fragments, 3)
        y = self.output_linear(y)
        y = self.dropout2(y)
        y = self.activation2(y)

        # Split y into mzs, probs, and flags
        mzs = y[:, :, 0]    # Shape: (batch_size, max_fragments)
        probs = y[:, :, 1]  # Shape: (batch_size, max_fragments)
        flags = y[:, :, 2]  # Shape: (batch_size, max_fragments)

        # Apply sigmoid to flags to get values between 0 and 1
        flags = torch.sigmoid(flags)

        # Multiply mzs by flags
        mzs = mzs * flags

        # Adjust probs where flags are zero
        probs = probs + torch.log(flags + 1e-6)  # Add a small value to avoid log(0)

        # Apply softmax to probs along the fragment dimension
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

def calculate_loss(predictions, actual_labels, sigma):
    pred_mzs, pred_probabilities, flag = predictions
    actual_mzs, actual_intensities = actual_labels
    
    print("pred_mzs shape:", pred_mzs.shape)
    print("actual_mzs shape:", actual_mzs.shape)
    print("pred_probabilities shape:", pred_probabilities.shape)
    print("actual_intensities shape:", actual_intensities.shape)
    
    # normalise actual_intensities by dividing by the sum along dim 1
    actual_probabilities = actual_intensities / torch.sum(actual_intensities, dim=1, keepdim=True)

    return gaussian_cross_entropy_loss(actual_mzs, actual_probabilities, pred_mzs, pred_probabilities, sigma)


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
        self.steps = 0
        # Get hidden size from the ChemBERTa model configuration
        self.hidden_size = self.model.config.hidden_size

        # Dimension of the supplementary data
        self.dim_supplementary_data = supplementary_data_dim

        # We'll initialize final_layers in the forward method
        self.final_layers = FinalLayers(self.hidden_size, self.max_seq_length, self.dim_supplementary_data, self.max_fragments, num_heads=8)

    def forward(self, input_ids, attention_mask, supplementary_data, labels):
        # Pass inputs through ChemBERTa
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)

        # Extract last hidden state (embeddings)
        last_hidden_state = outputs.hidden_states[-1]  

        predicted_output = self.final_layers(last_hidden_state, supplementary_data, attention_mask)  # Shape: [batch_size, max_fragments, 3]
        
        # Schedule sigma from 1.0 to 0.0001 over the course of 100000 steps
        sigma = 1.0 - (1.0 - 0.0001) * (self.steps / 100000)
        self.steps += 1
        
        # Calculate the loss by comparing to labels
        loss = calculate_loss(predicted_output, labels, sigma=sigma)

        return loss, predicted_output
    
    def evaluate_spectra(self, predicted_output, interleaved_labels, threshold=0.5):
        """
        Evaluate spectra using greedy cosine and hungarian cosine metrics, processing both 
        the predicted output and the interleaved labels to extract m/z values and intensities.
        
        Parameters:
        - predicted_output: Tuple (pred_mz, pred_probs, pred_flags)
        - interleaved_labels: Ground truth in interleaved format (m/z and intensities)
        - threshold: A cutoff value for flags to zero out certain predictions.
        
        Returns:
        - A dictionary with the greedy and Hungarian cosine scores.
        """

        # Step 1: Process the predicted output
        pred_mz, pred_intensities = process_predicted_output(predicted_output, threshold)

        # Step 2: Extract the ground truth m/z and intensities from interleaved labels
        mz_true, intensities_true = extract_fragments_from_interleaved(interleaved_labels)

        # Step 3: Calculate the metrics using the processed predicted and ground truth values
        greedy_score = greedy_cosine(pred_mz, mz_true, pred_intensities, intensities_true)
        #hungarian_score = hungarian_cosine(pred_mz, mz_true, pred_intensities, intensities_true)

        return {
            'greedy_cosine': greedy_score
            #'hungarian_cosine': hungarian_score
        }    


def process_predicted_output(predicted_output):
    """
    Processes the predicted output to extract m/z values and intensities.
    
    Parameters:
    - predicted_output: Tuple containing predicted m/z, predicted probabilities (for intensities), and flags
    - threshold: A cutoff value for the flags. If flag > threshold, the corresponding m/z and intensity are set to zero.
    
    Returns:
    - pred_mz: Processed predicted m/z values
    - pred_intensities: Normalized predicted intensities
    """
    pred_mz, pred_probs, pred_flags = predicted_output

    # Apply the threshold: if flag > 0.5, keep values, else zero them out
    mask = (pred_flags > 0.5).float()

    # Set values to zero where the flag is below the threshold
    pred_mz = pred_mz * mask
    pred_probs = pred_probs * mask

    # Renormalize the predicted intensities (pred_probs) so the maximum is 1
    max_intensity = torch.max(pred_probs, dim=1, keepdim=True).values
    pred_intensities = torch.where(max_intensity > 0, pred_probs / max_intensity, torch.zeros_like(pred_probs))
    
    return pred_mz, pred_intensities


def create_spectrum(mz, intensities):
    """
    Create a Spectrum object from m/z values and intensities for use in matchms.
    
    Parameters:
    - mz: m/z values
    - intensities: Intensity values
    
    Returns:
    - spectrum: A matchms Spectrum object
    """
    metadata = {}  # You can populate this with any relevant metadata if necessary
    return Spectrum(mz=mz.cpu().numpy(), intensities=intensities.cpu().numpy(), metadata=metadata)

def greedy_cosine(mz_a, mz_b, intensities_a, intensities_b):
    """
    Use CosineGreedy from matchms to calculate greedy cosine similarity between two spectra.
    
    Parameters:
    - mz_a: m/z values for spectrum A
    - mz_b: m/z values for spectrum B
    - intensities_a: Intensity values for spectrum A
    - intensities_b: Intensity values for spectrum B
    
    Returns:
    - similarity: CosineGreedy similarity score
    """
    # Create Spectrum objects
    spectrum_a = create_spectrum(mz_a, intensities_a)
    spectrum_b = create_spectrum(mz_b, intensities_b)

    # Instantiate the CosineGreedy similarity function
    cosine_greedy = CosineGreedy(tolerance=0.1)  # Adjust tolerance as needed

    # Compute the similarity score
    score, _ = cosine_greedy.pair(spectrum_a, spectrum_b)
    return score

def hungarian_cosine(mz_a, mz_b, intensities_a, intensities_b):
    """
    Use CosineHungarian from matchms to calculate Hungarian cosine similarity between two spectra.
    
    Parameters:
    - mz_a: m/z values for spectrum A
    - mz_b: m/z values for spectrum B
    - intensities_a: Intensity values for spectrum A
    - intensities_b: Intensity values for spectrum B
    
    Returns:
    - similarity: CosineHungarian similarity score
    """
    # Create Spectrum objects
    spectrum_a = create_spectrum(mz_a, intensities_a)
    spectrum_b = create_spectrum(mz_b, intensities_b)

    # Instantiate the CosineHungarian similarity function
    cosine_hungarian = CosineHungarian(tolerance=0.1)  # Adjust tolerance as needed

    # Compute the similarity score
    score, _ = cosine_hungarian.pair(spectrum_a, spectrum_b)
    return score



if __name__ == "__main__":
    # Example interleaved lists (padded and converted to tensors)

    model = AutoModel.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
    
    # Add the supplementary_data_dim parameter
    MS_model = CustomChemBERTaModel(model, max_fragments=512, max_seq_length=512, supplementary_data_dim=75)  # Adjust the value of supplementary_data_dim as needed

    # print(MS_model)

    # for name, param in MS_model.named_parameters():
    #     if param.requires_grad:
    #         print(f"{name} has shape {param.shape}")
    
    pred_vect = torch.tensor([[338, 0.3, 1, 350, 0.01, 1, 253, 1, 1, 0, 0, 0, 0, 0, 0], 
                              [350, 0.01, 1, 253, 1, 1, 338, 0.3, 1, 0, 0, 0, 0, 0, 0], 
                              [0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0]])
    actual_vect = torch.tensor([[338, 0.3, 1, 350, 0.01, 1, 253, 1, 1, 0, 0, 0, 0, 0, 0], 
                              [350, 0.01, 1, 253, 1, 1, 338, 0.3, 1, 0, 0, 0, 0, 0, 0]])
    

    loss_value = calculate_loss(pred_vect, actual_vect, sigma=0.1)
    print(f"Loss: {loss_value}")

    


