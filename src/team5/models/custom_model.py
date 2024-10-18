import logging

import torch.nn as nn
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch.nn.functional as F
from matchms import Spectrum
from matchms.similarity import CosineGreedy
import numpy as np

logging.getLogger("matchms").setLevel(logging.ERROR)

class FinalLayers(nn.Module):
    def __init__(self, hidden_size, supplementary_data_dim, num_bins):
        super(FinalLayers, self).__init__()
        # Process supplementary data
        self.supp_layer = nn.Sequential(
            nn.Linear(supplementary_data_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # Main processing of tokens 
        self.main_layers_tokens = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1)
        )


        # Project to a smaller dimension
        self.project_hidden = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.1))
        
        # Output layers
        self.output_layers = nn.Sequential(
            nn.Linear(512, num_bins),
            nn.LayerNorm(num_bins),
            nn.ReLU()
        )

    def forward(self, x, supplementary_data, attention_mask):
        # Process supplementary data
        supp = self.supp_layer(supplementary_data).unsqueeze(1)
        
        # Combine with input and apply mask
        x = x * attention_mask.unsqueeze(-1) + supp
        
        # Main processing of tokens
        x = x + self.main_layers_tokens(x)  # Residual connection

        # Mean pooling of tokens
        x = x.mean(dim=1)

        # Project to a smaller dimension
        x = self.project_hidden(x) 

        # Output layers
        x = self.output_layers(x)

        return x




class CustomChemBERTaModel(nn.Module):
    def __init__(self, model, max_fragments, max_seq_length, supplementary_data_dim, 
                 max_mz=2000, delta_mz=0.1, intensity_power=0.5):
        super(CustomChemBERTaModel, self).__init__()
        self.model = model
        self.max_fragments = max_fragments
        self.max_seq_length = max_seq_length
        self.hidden_size = self.model.config.hidden_size
        self.dim_supplementary_data = supplementary_data_dim
        self.max_mz = max_mz
        self.delta_mz = delta_mz
        self.intensity_power = intensity_power
        self.num_bins = int(max_mz / delta_mz)
        
        self.mz_weights = torch.linspace(0, 1, self.num_bins)
        
        self.final_layers = FinalLayers(self.hidden_size, 
                                        self.dim_supplementary_data, 
                                        self.num_bins)
        
        # Training mode flag
        self.training_mode = True

    def forward(self, input_ids, attention_mask=None, supplementary_data=None, labels=None):
        # get the last hidden state from the model
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        last_hidden_state = outputs.hidden_states[-1]  

        # Use the last hidden state as the input to the final layers
        predicted_output = self.final_layers(last_hidden_state, supplementary_data, attention_mask)
        
        if labels is not None:
            loss = self.calculate_loss(predicted_output, labels)
            return loss, predicted_output
        else:
            return predicted_output
        
    
    def extract_from_actual_labels(self, actual_labels):
        actual_mzs = actual_labels[:, :, 0]  # Shape: (batch_size, max_fragments)
        actual_intensities = actual_labels[:, :, 1]  # Shape: (batch_size, max_fragments)
        return actual_mzs, actual_intensities

    def calculate_loss(self, predictions, actual_labels):
        pred_intensities = predictions  # Shape: (batch_size, num_bins)
        actual_mzs, actual_intensities = self.extract_from_actual_labels(actual_labels)
        
        print(f"Predictions shape: {pred_intensities.shape}")
        print(f"Actual m/z shape: {actual_mzs.shape}")
        print(f"Actual intensities shape: {actual_intensities.shape}")
        
        # Bin the actual intensities
        binned_actual = self.bin_intensities(actual_mzs, actual_intensities)
        
        print(f"Binned actual shape: {binned_actual.shape}")
        
        # Apply intensity power
        pred_intensities_pow = pred_intensities ** self.intensity_power
        binned_actual_pow = binned_actual ** self.intensity_power
        
        # Normalize
        pred_intensities_norm = pred_intensities_pow / pred_intensities_pow.sum(dim=1, keepdim=True).clamp(min=1e-8)
        binned_actual_norm = binned_actual_pow / binned_actual_pow.sum(dim=1, keepdim=True).clamp(min=1e-8)
        
        # Ensure mz_weights is the correct shape
        mz_weights = self.mz_weights.to(pred_intensities.device)
        if mz_weights.shape[0] != pred_intensities.shape[1]:
            mz_weights = torch.linspace(0, 1, pred_intensities.shape[1], device=pred_intensities.device)
        
        # Compute loss
        loss = ((pred_intensities_norm - binned_actual_norm).square() * mz_weights).sum(dim=1).mean()
        
        return loss

    def bin_intensities(self, mzs, intensities):
        bin_indices = (mzs / self.delta_mz).long().clamp(min=0, max=self.num_bins-1)
        binned = torch.zeros(intensities.shape[0], self.num_bins, device=intensities.device)
        binned.scatter_add_(1, bin_indices, intensities)
        return binned


    def train(self, mode=True):
        super(CustomChemBERTaModel, self).train(mode)
        self.training_mode = mode

    def eval(self):
        super(CustomChemBERTaModel, self).eval()
        self.training_mode = False

    def reset_steps(self):
        self.steps = 0
    
    def evaluate_spectra(self, predicted_output, labels):
        pred_intensities = predicted_output
        actual_mzs, actual_intensities = self.extract_from_actual_labels(labels)
        
        # Convert binned predictions back to discrete peaks
        pred_mzs = torch.linspace(0, self.max_mz, self.num_bins, device=pred_intensities.device)
        pred_mzs = pred_mzs.unsqueeze(0).expand(pred_intensities.shape[0], -1)
        
        # Find top K peaks
        K = min(self.max_fragments, self.num_bins)
        top_k_values, top_k_indices = torch.topk(pred_intensities, k=K, dim=1)
        pred_mzs = torch.gather(pred_mzs, 1, top_k_indices)
        pred_intensities = top_k_values
        
        # Compute greedy cosine similarity
        greedy_scores = []
        for i in range(pred_mzs.shape[0]):
            greedy_score = greedy_cosine(pred_mzs[i], actual_mzs[i], pred_intensities[i], actual_intensities[i])
            greedy_scores.append(greedy_score)
        greedy_score = sum(greedy_scores) / len(greedy_scores)
        
        return {'greedy_score': greedy_score}


    def process_predicted_output(self, predicted_output, probability_cutoff=0.01):
        """
        Processes the predicted output to extract m/z values and intensities,
        applying a probability cutoff.

        Parameters:
        - predicted_output: Tensor of shape (batch_size, num_bins) containing binned intensities
        - probability_cutoff: Minimum probability to keep a peak (default: 0.01)

        Returns:
        - pred_mz: Processed predicted m/z values
        - pred_intensities: Normalized predicted intensities
        """
        pred_intensities = predicted_output

        # Create m/z values corresponding to bin centers
        pred_mz = torch.linspace(0, self.max_mz, self.num_bins, device=pred_intensities.device)
        pred_mz = pred_mz.unsqueeze(0).expand(pred_intensities.shape[0], -1)

        # Normalize intensities
        max_intensity = torch.max(pred_intensities, dim=1, keepdim=True).values
        pred_intensities = torch.where(max_intensity > 0, 
                                    pred_intensities / max_intensity, 
                                    torch.zeros_like(pred_intensities))

        # Apply probability cutoff
        mask = pred_intensities >= probability_cutoff
        pred_mz = pred_mz * mask
        pred_intensities = pred_intensities * mask

        # Remove zero entries
        non_zero_mask = pred_intensities.sum(dim=1) != 0
        pred_mz = [mz[mask] for mz, mask in zip(pred_mz, non_zero_mask)]
        pred_intensities = [intensities[mask] for intensities, mask in zip(pred_intensities, non_zero_mask)]

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
    print(f"shape of input_encodings: {input_encodings['input_ids'].shape}")

    input_ids = input_encodings["input_ids"]  # Token IDs
    attention_mask = input_encodings["attention_mask"]  # Attention mask
    supplementary_data = torch.randn(batch_size, SUPPLEMENTARY_DATA_DIM)  # Random supplementary data

    # Create labels tensor
    mz_values = torch.randint(1, int(custom_model.max_mz), (batch_size, MAX_FRAGMENTS))
    intensities = torch.rand((batch_size, MAX_FRAGMENTS))
    labels = torch.stack([mz_values, intensities], dim=-1)

    print(f"labels shape: {labels.shape}")

    # Forward pass
    with torch.no_grad():
        loss, predicted_output = custom_model(input_ids, attention_mask, supplementary_data, labels=labels)
        print(f"predicted_output shape: {predicted_output.shape}")

    # Calculate loss
    loss = custom_model.calculate_loss(predicted_output, labels)
    print(f"Loss value: {loss.item()}")

    # Evaluate spectra
    evaluation_score = custom_model.evaluate_spectra(predicted_output, labels)
    print(f"Evaluation score: {evaluation_score}")

    # After your existing test code
    perfect_pred = custom_model.bin_intensities(labels[:,:,0], labels[:,:,1])
    perfect_loss = custom_model.calculate_loss(perfect_pred, labels)
    print(f"Loss for perfect prediction: {perfect_loss.item()}")

    def test_simple_cases(custom_model):
        # Set up a simple scenario with 3 peaks
        batch_size = 1
        num_peaks = 3
        max_mz = 1000
        delta_mz = 1  # 1 Da bins for simplicity

        custom_model.max_mz = max_mz
        custom_model.delta_mz = delta_mz
        custom_model.num_bins = int(max_mz / delta_mz)
        custom_model.mz_weights = torch.linspace(0, 1, custom_model.num_bins)

        # Case 1: Perfect match
        mz_values = torch.tensor([[100, 500, 900]])
        intensities = torch.tensor([[0.2, 0.5, 0.3]])
        labels = torch.stack([mz_values, intensities], dim=-1)
        
        pred_intensities = custom_model.bin_intensities(mz_values, intensities)
        loss = custom_model.calculate_loss(pred_intensities, labels)
        print(f"Loss for perfect match: {loss.item()}")

        # Case 2: Slight mismatch
        pred_intensities_slight_mismatch = pred_intensities.clone()
        pred_intensities_slight_mismatch[0, 100] = 0.25  # Change 0.2 to 0.25
        pred_intensities_slight_mismatch[0, 500] = 0.45  # Change 0.5 to 0.45
        loss = custom_model.calculate_loss(pred_intensities_slight_mismatch, labels)
        print(f"Loss for slight mismatch: {loss.item()}")

        # Case 3: Large mismatch
        pred_intensities_large_mismatch = torch.zeros_like(pred_intensities)
        pred_intensities_large_mismatch[0, [200, 600, 800]] = torch.tensor([0.3, 0.4, 0.3])
        loss = custom_model.calculate_loss(pred_intensities_large_mismatch, labels)
        print(f"Loss for large mismatch: {loss.item()}")

        # Case 4: All intensity in one wrong bin
        pred_intensities_one_bin = torch.zeros_like(pred_intensities)
        pred_intensities_one_bin[0, 300] = 1.0
        loss = custom_model.calculate_loss(pred_intensities_one_bin, labels)
        print(f"Loss for all intensity in one wrong bin: {loss.item()}")

    # Call the function in your main testing section
    test_simple_cases(custom_model)

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


