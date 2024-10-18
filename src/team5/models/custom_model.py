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
        
        # Bin the actual intensities
        binned_actual = self.bin_intensities(actual_mzs, actual_intensities)
        
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



