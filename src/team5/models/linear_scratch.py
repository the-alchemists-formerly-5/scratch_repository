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
    def __init__(self, hidden_size, supplementary_data_dim, max_fragments):
        super(FinalLayers, self).__init__()

        self.max_fragments = max_fragments

        # Process supplementary data
        self.supp_layer = nn.Sequential(
            nn.Linear(supplementary_data_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # Main processing layers
        self.main_layers = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # Output layers
        self.mz_output = nn.Sequential(
            nn.Linear(hidden_size, max_fragments),
            nn.ReLU()  # Ensure non-negative m/z values
        )
        self.prob_output = nn.Sequential(
            nn.Linear(hidden_size, max_fragments),
            nn.Sigmoid()  # Ensure probabilities are between 0 and 1
        )

    def forward(self, x, supplementary_data, attention_mask):
        # Process supplementary data
        supp = self.supp_layer(supplementary_data).unsqueeze(1)
        
        # Combine with input and apply mask
        x = x * attention_mask.unsqueeze(-1) + supp
        
        # Main processing
        x = x + self.main_layers(x)  # Residual connection
        
        # Output
        mzs = self.mz_output(x.mean(dim=1))  # Global average pooling
        probs = self.prob_output(x.mean(dim=1))
        
        return torch.stack([mzs, probs], dim=-1)




class CustomChemBERTaModel(nn.Module):
    def __init__(self, model, max_fragments, max_seq_length, supplementary_data_dim, 
                 initial_sigma=1.0, final_sigma=1.0, eval_sigma=1.0, total_steps=1000000):
        super(CustomChemBERTaModel, self).__init__()
        self.model = model
        self.max_fragments = max_fragments
        self.max_seq_length = max_seq_length
        self.steps = 0
        self.hidden_size = self.model.config.hidden_size
        self.dim_supplementary_data = supplementary_data_dim
        self.final_layers = FinalLayers(self.hidden_size, 
                                        self.dim_supplementary_data, self.max_fragments)
        
        # Sigma scheduling parameters
        self.initial_sigma = initial_sigma
        self.final_sigma = final_sigma
        self.eval_sigma = eval_sigma
        self.total_steps = total_steps
        
        # Training mode flag
        self.training_mode = True

    def forward(self, input_ids, attention_mask=None, supplementary_data=None, labels=None, **kwargs):
        # get the last hidden state from the model
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        last_hidden_state = outputs.hidden_states[-1]  

        # Use the last hidden state as the input to the final layers
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
        #pred_flags = predicted_output[:, :, 2]  # Predicted flags (already used in mzs and probs calculations)
        #return pred_mzs, pred_probabilities, pred_flags
        return pred_mzs, pred_probabilities
    
    def extract_from_actual_labels(self, actual_labels):
        actual_mzs = actual_labels[:, :, 0]  # Shape: (batch_size, max_fragments)
        actual_intensities = actual_labels[:, :, 1]  # Shape: (batch_size, max_fragments)
        actual_probabilities = actual_intensities / (torch.sum(actual_intensities, dim=1, keepdim=True) + 1e-8)
        return actual_mzs, actual_intensities, actual_probabilities

    def calculate_loss(self, predictions, actual_labels, sigma):

        #pred_mzs, pred_probabilities, pred_flags = self.extract_from_predicted_output(predictions)
        pred_mzs, pred_probabilities= self.extract_from_predicted_output(predictions)
        actual_mzs, actual_intensities, actual_probabilities = self.extract_from_actual_labels(actual_labels)
        return torch.abs(self.gaussian_kl_loss(pred_mzs, pred_probabilities, actual_mzs, actual_probabilities, sigma=sigma))

    def gaussian_kl_loss(self, predicted_mzs, predicted_probabilities, actual_mzs, actual_probabilities, sigma):
        """
        Compute KL divergence loss using Gaussian soft binning for batched input.
        
        :param actual_mzs: Tensor of shape [batch_size, n_actual_peaks] - true m/z values
        :param actual_probabilities: Tensor of shape [batch_size, n_actual_peaks] - true probabilities
        :param predicted_mzs: Tensor of shape [batch_size, n_predicted_peaks] - predicted m/z values
        :param predicted_probabilities: Tensor of shape [batch_size, n_predicted_peaks] - predicted probabilities
        :param sigma: Float - standard deviation for Gaussian kernel
        :return: KL divergence loss
        """

        def soft_binning_custom_unnormalized(values, probs, actual_mzs, sigma=1):
            """
            Apply soft binning to a batch of distributions.
            
            :param values: Tensor of shape [batch_size, n_peaks] - m/z values
            :param probs: Tensor of shape [batch_size, n_peaks] - probabilities
            :param actual_mzs: Tensor of shape [batch_size, n_centers] - bin centers (actual m/z values)
            :param sigma: Float - standard deviation for Gaussian kernel
            :return: Tensor of shape [batch_size, n_centers] - soft-binned probabilities
            """
            # Compute differences: [batch_size, n_peaks, n_centers]
            diff = values.unsqueeze(2) - actual_mzs.unsqueeze(1)
            
            # Compute Gaussian weights: [batch_size, n_peaks, n_centers]
            weight = torch.exp(-0.5 * (diff / sigma)**2)
            
            # Apply weights to probabilities and sum: [batch_size, n_centers]
            return (weight * probs.unsqueeze(2)).sum(dim=1)

        # Apply soft binning to predicted distribution
        Q = soft_binning_custom_unnormalized(predicted_mzs, predicted_probabilities, actual_mzs, sigma)
        
        # Clamp values to avoid log(0) issues
        Q = Q.clamp(1e-10)
        
        # Calculate KL divergence
        # actual_probabilities are used directly as they align with their own centers
        kl_div = F.kl_div(Q.log(), actual_probabilities, reduction='none')
        
        # Sum over peaks and average over batch
        return kl_div.sum(dim=1).mean()
    
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
        #pred_mz, pred_probs, pred_flags = self.extract_from_predicted_output(predicted_output)
        pred_mz, pred_probs = self.extract_from_predicted_output(predicted_output)
        # Apply the threshold: if flag > 0.5, keep values, else zero them out
        #mask = (pred_flags > 0.5).float()

        # Set values to zero where the flag is below the threshold
        #pred_mz = pred_mz * mask
        #pred_probs = pred_probs * mask

        # Step 2: Extract the ground truth m/z and intensities from labels
        mz_true, intensities_true, probabilities_true = self.extract_from_actual_labels(labels)

        # Step 3: Calculate the metrics using the processed predicted and ground truth values, for each sample in the batch, then average
        greedy_scores = []
        for i in range(pred_mz.shape[0]):
            greedy_score = greedy_cosine(pred_mz[i], mz_true[i], pred_probs[i], probabilities_true[i])
            greedy_scores.append(greedy_score)
        greedy_score = sum(greedy_scores) / len(greedy_scores)
        #hungarian_score = hungarian_cosine(pred_mz, mz_true, pred_intensities, intensities_true)

        return {
            'greedy_score': greedy_score,
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
    #pred_mz, pred_probs, pred_flags = predicted_output
    pred_mz, pred_probs = predicted_output
    # Apply the threshold: if flag > 0.5, keep values, else zero them out
    #mask = (pred_flags > 0.5).float()

    # Set values to zero where the flag is below the threshold
    #pred_mz = pred_mz * mask
    #pred_probs = pred_probs * mask

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
    with torch.no_grad():
        loss, predicted_output = custom_model(input_ids, attention_mask, supplementary_data, labels=labels)
        print(f"predicted_output shape: {predicted_output.shape}")

   # Create labels tensor
    mz_values = torch.randint(100, 401, (batch_size, MAX_FRAGMENTS))
    intensities = torch.rand((batch_size, MAX_FRAGMENTS))
    probabilities = intensities / intensities.sum(dim=1, keepdim=True)  # Normalize to sum to 1 for each batch
    labels = torch.stack([mz_values, probabilities], dim=-1)

    print(f"labels shape: {labels.shape}")

    # Create pred_output tensor
    pred_mzs = mz_values
    pred_probabilities = probabilities
    pred_flags = torch.ones((batch_size, MAX_FRAGMENTS))  # All values set to 1
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