import logging

import torch.nn as nn
import torch
from torch.nn.functional import cosine_similarity
from transformers import AutoModelForMaskedLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
import torch.nn.functional as F
from matchms import Spectrum
from matchms.similarity import CosineGreedy, CosineHungarian
import numpy as np

logging.getLogger("matchms").setLevel(logging.ERROR)

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

        # Softmax on probs
        probs = F.softmax(probs, dim=1)

        # Stack together mzs, probs, and flags
        output = torch.stack([mzs, probs, flags], dim=-1)

        return output




class CustomChemBERTaModel(nn.Module):
    def __init__(self, model, max_fragments, max_seq_length, supplementary_data_dim, 
                 initial_sigma=2.0, final_sigma=0.0001, eval_sigma=0.1, total_steps=1000000):
        super(CustomChemBERTaModel, self).__init__()
        self.model = model
        self.max_fragments = max_fragments
        self.max_seq_length = max_seq_length
        self.steps = 0
        self.hidden_size = self.model.config.hidden_size
        self.dim_supplementary_data = supplementary_data_dim
        self.final_layers = FinalLayers(self.hidden_size, self.max_seq_length, 
                                        self.dim_supplementary_data, self.max_fragments, num_heads=8)
        
        # Sigma scheduling parameters
        self.initial_sigma = initial_sigma
        self.final_sigma = final_sigma
        self.eval_sigma = eval_sigma
        self.total_steps = total_steps
        
        # Training mode flag
        self.training_mode = True

    def forward(self, input_ids, attention_mask, supplementary_data, labels=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        last_hidden_state = outputs.hidden_states[-1]  
        predicted_output = self.final_layers(last_hidden_state, supplementary_data, attention_mask)
        
        if self.training_mode:
            # Schedule sigma only during training
            sigma = self.initial_sigma - (self.initial_sigma - self.final_sigma) * (self.steps / self.total_steps)
            self.steps += 1
        else:
            # Use final sigma during evaluation
            sigma = self.eval_sigma
        
        if labels is not None:
            loss = self.calculate_loss(predicted_output, labels, sigma=sigma)
            return loss, predicted_output
        else:
            return predicted_output
        
    def extract_from_predicted_output(self, predicted_output):
        pred_mzs = predicted_output[:, :, 0]    # Predicted m/z values
        pred_probabilities = predicted_output[:, :, 1]  # Predicted probabilities
        pred_flags = predicted_output[:, :, 2]  # Predicted flags (already used in mzs and probs calculations)
        return pred_mzs, pred_probabilities, pred_flags
    
    def extract_from_actual_labels(self, actual_labels):
        actual_mzs = actual_labels[:, :, 0]  # Shape: (batch_size, max_fragments)
        actual_intensities = actual_labels[:, :, 1]  # Shape: (batch_size, max_fragments)
        actual_probabilities = actual_intensities / (torch.sum(actual_intensities, dim=1, keepdim=True) + 1e-8)
        return actual_mzs, actual_intensities, actual_probabilities

    def calculate_loss(self, predictions, actual_labels, sigma):

        pred_mzs, pred_probabilities, pred_flags = self.extract_from_predicted_output(predictions)
        actual_mzs, actual_intensities, actual_probabilities = self.extract_from_actual_labels(actual_labels)
        return self.gaussian_cosine_loss(pred_mzs, pred_probabilities, actual_mzs, actual_probabilities, sigma=sigma)

    def gaussian_cosine_loss(self, pred_mzs, pred_probabilities, actual_mzs, actual_probabilities, sigma, epsilon=1e-10):

        print(f"pred_probabilities:\n {pred_probabilities}")
        print(f"actual_probabilities:\n {actual_probabilities}")

        def gaussian_prediction(x, centers, heights, sigma):
            x = x.unsqueeze(2)
            centers = centers.unsqueeze(1)
            heights = heights.unsqueeze(1)
            
            gauss = torch.exp(-((x - centers) ** 2) / (2 * sigma ** 2))
            
            result = torch.sum(heights * gauss, dim=2)
            print(f"result: \n {result}")
            return result

        # Compute gaussian predictions for actual mzs
        gaussian_pred_probabilities = gaussian_prediction(actual_mzs, pred_mzs, pred_probabilities, sigma)

        # Clamp values to avoid zeros
        gaussian_pred_probabilities = torch.clamp(gaussian_pred_probabilities, min=epsilon)

        # Compute cosine similarity
        similarity = F.cosine_similarity(gaussian_pred_probabilities, actual_probabilities, dim=1)

        # Convert similarity to loss (1 - similarity)
        loss = 1 - similarity

        return loss.mean()

    def train(self, mode=True):
        super(CustomChemBERTaModel, self).train(mode)
        self.training_mode = mode

    def eval(self):
        super(CustomChemBERTaModel, self).eval()
        self.training_mode = False

    def reset_steps(self):
        self.steps = 0
    
    def evaluate_spectra(self, predicted_output, labels):
        """
        Evaluate spectra using greedy cosine and hungarian cosine metrics, processing both 
        the predicted output and the labels to extract m/z values and intensities.
        
        Parameters:
        - predicted_output: Tuple (pred_mz, pred_probs, pred_flags)
        - labels: Ground truth labels Tuple(m/z, intensities)
        
        Returns:
        - A dictionary with the greedy cosine score.
        """

        # Step 1: Process the predicted output
        pred_mz, pred_probs, pred_flags = self.extract_from_predicted_output(predicted_output)
        # Apply the threshold: if flag > 0.5, keep values, else zero them out
        mask = (pred_flags > 0.5).float()

        # Set values to zero where the flag is below the threshold
        pred_mz = pred_mz * mask
        pred_probs = pred_probs * mask

        # Step 2: Extract the ground truth m/z and intensities from labels
        mz_true, intensities_true, probabilities_true = self.extract_from_actual_labels(labels)

        # Step 3: Calculate the metrics using the processed predicted and ground truth values, for each sample in the batch, then average
        greedy_scores = []
        for i in range(pred_mz.shape[0]):
            greedy_score = greedy_cosine(pred_mz[i], mz_true[i], pred_probs[i], probabilities_true[i])
            greedy_scores.append(greedy_score)
        greedy_score = sum(greedy_scores) / len(greedy_scores)
        #hungarian_score = hungarian_cosine(pred_mz, mz_true, pred_intensities, intensities_true)

        return greedy_score


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
    metadata = {}
    mz = mz.detach().numpy()
    intensities = intensities.detach().numpy()
    # sort the mz and intensities, according to mz
    sorted_indices = np.argsort(mz)
    mz = mz[sorted_indices]
    intensities = intensities[sorted_indices]
    return Spectrum(mz=mz, intensities=intensities, metadata=metadata)

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
    try:
        result = cosine_greedy.pair(spectrum_a, spectrum_b)
        score, _ = result.tolist()
    except ZeroDivisionError as e:
        score = 1.
    return score



# ----- Testing -----


# if __name__ == "__main__":
    

#     model = AutoModel.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
    
#     # Add the supplementary_data_dim parameter
#     MS_model = CustomChemBERTaModel(model, max_fragments=512, max_seq_length=512, supplementary_data_dim=81)  # Adjust the value of supplementary_data_dim as needed

#     print(MS_model)

#     for name, param in MS_model.named_parameters():
#         if param.requires_grad:
#             print(f"{name} has shape {param.shape}")
    
if __name__ == "__main__":
    print("Testing")

    # Load the real ChemBERTa model and tokenizer
    chemBERTa_model = AutoModelForMaskedLM.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
    tokenizer = AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
    
    BASE_MODEL = "seyonec/ChemBERTa-zinc-base-v1"
    MAX_FRAGMENTS = 4 # from anton, max number of mzs/intensities
    MAX_SEQ_LENGTH = 10 # base model max seq length
    SUPPLEMENTARY_DATA_DIM = 81
    batch_size = 8  # Ensure this is set to 8

    # Initialize the CustomChemBERTaModel (untrained)
    custom_model = CustomChemBERTaModel(chemBERTa_model, MAX_FRAGMENTS, MAX_SEQ_LENGTH, SUPPLEMENTARY_DATA_DIM)

    # Create dummy input data
    input_smiles = ["CCONaCO", "CCC", "CCN", "CCCl", "CCBr", "CCCC", "CCCCO", "CCNCl"]
    input_encodings = tokenizer(
        input_smiles, 
        padding='max_length',  # Ensure padding to max_seq_length
        truncation=True, 
        return_tensors="pt", 
        max_length=MAX_SEQ_LENGTH  # Explicitly define max sequence length
    )
    #print(f"input_encodings: {input_encodings}")
    print(f" shape of input_encodings: {input_encodings['input_ids'].shape}")

    input_ids = input_encodings["input_ids"]  # Token IDs
    attention_mask = input_encodings["attention_mask"]  # Attention mask
    supplementary_data = torch.randn(batch_size, SUPPLEMENTARY_DATA_DIM)  # Random supplementary data
    labels = torch.randn(batch_size, MAX_FRAGMENTS, 2)  # Random actual m/z and intensities for testing


   # Create labels tensor
    mz_values = torch.randint(100, 401, (batch_size, MAX_FRAGMENTS))
    intensities = torch.rand((batch_size, MAX_FRAGMENTS))
    probabilities = intensities / intensities.sum(dim=1, keepdim=True)  # Normalize to sum to 1 for each batch
    labels = torch.stack([mz_values, probabilities], dim=-1)

    print(f"labels shape: {labels.shape}")

    # Create pred_output tensor
    pred_mzs = mz_values
    pred_probabilities = probabilities
    pred_flags = torch.rand((batch_size, MAX_FRAGMENTS))  # Random values between 0 and 1
    pred_output = torch.stack([pred_mzs, pred_probabilities, pred_flags], dim=-1)

    print(f"pred_output shape: {pred_output.shape}")

    # Calculate loss
    sigma = 0.00001
    loss = custom_model.calculate_loss(pred_output, labels, sigma=sigma)
    print(f"Loss value: {loss.item()}")

    # Evaluate spectra
    evaluation_score = custom_model.evaluate_spectra(pred_output, labels)
    print(f"Evaluation score: {evaluation_score}")


    # # Test the loss calculation
    # print(f"Loss value: {loss.item()}")

    # # Verify the shape of the predicted output
    # assert pred_output.shape == (batch_size, max_fragments, 3), f"Expected shape {(batch_size, max_fragments, 3)}, but got {pred_output.shape}"
    # print(f"Output shape is correct: {pred_output.shape}")

    # # Now test the evaluate_spectra function with dummy data
    # predicted_mzs = pred_output[:, :, 0]  # Extract predicted m/z values
    # predicted_probs = pred_output[:, :, 1]  # Extract predicted probabilities (intensities)
    # predicted_flags = pred_output[:, :, 2]  # Extract predicted flags
    
    # # Extract ground truth m/z and intensities for evaluation
    # actual_mzs = labels[:, :, 0]
    # actual_intensities = labels[:, :, 1]

    # # Package predicted outputs and labels for the evaluate_spectra function
    # predicted_output = (predicted_mzs, predicted_probs, predicted_flags)
    # labels_for_eval = (actual_mzs, actual_intensities)

    # # Perform spectra evaluation
    # evaluation_results = MS_model.evaluate_spectra(predicted_output, labels_for_eval)
    # print(f"Greedy cosine score: {evaluation_results['greedy_cosine']}")

