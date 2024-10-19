import logging
import torch.nn as nn
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch.nn.functional as F
from matchms import Spectrum
from matchms.similarity import CosineGreedy
import numpy as np

logging.getLogger("matchms").setLevel(logging.ERROR)

class ClampedReLU(nn.Module):
    def __init__(self, max_value):
        super().__init__()
        self.max_value = max_value

    def forward(self, x):
        return torch.clamp(F.relu(x), max=self.max_value)

class FinalLayers(nn.Module):
    def __init__(self, hidden_size, supplementary_data_dim, max_fragments, mz_max_value=2000, prob_threshold=1e-4):
        super(FinalLayers, self).__init__()

        self.max_fragments = max_fragments
        self.hidden_size = hidden_size
        self.prob_threshold = prob_threshold

        # Linear layer to process supplementary data
        self.supplementary_linear = nn.Linear(supplementary_data_dim, hidden_size)
        self.supplementary_activation = nn.GELU()
        self.supplementary_dropout = nn.Dropout(0.1)
        self.supplementary_norm = nn.LayerNorm(hidden_size)

        # Attention pooling
        self.attention_query = nn.Linear(hidden_size, 1)

        # MLP after attention pooling
        self.mlp_fc1 = nn.Linear(hidden_size, hidden_size)
        self.mlp_fc2 = nn.Linear(hidden_size, hidden_size)
        self.mlp_activation = nn.GELU()
        self.mlp_dropout = nn.Dropout(0.1)
        self.mlp_norm = nn.LayerNorm(hidden_size)

        # Output linear layers
        self.output_linear_mz = nn.Linear(hidden_size, self.max_fragments)
        self.output_linear_prob = nn.Linear(hidden_size, self.max_fragments)
        
        # Activation for m/z values
        self.mz_activation = ClampedReLU(max_value=mz_max_value)

    def forward(self, x, supplementary_data, attention_mask):
        # x: (batch_size, seq_length, hidden_size)
        # supplementary_data: (batch_size, supplementary_data_dim)
        # attention_mask: (batch_size, seq_length)

        # Process supplementary data
        supplementary_data = self.supplementary_linear(supplementary_data)
        supplementary_data = self.supplementary_dropout(supplementary_data)
        supplementary_data = self.supplementary_norm(supplementary_data)
        supplementary_data = self.supplementary_activation(supplementary_data)

        # Attention pooling
        attention_scores = self.attention_query(x).squeeze(-1)  # (batch_size, seq_length)
        attention_scores = attention_scores.masked_fill(attention_mask == 0, float('-inf'))
        attention_weights = F.softmax(attention_scores, dim=-1)
        pooled = torch.bmm(attention_weights.unsqueeze(1), x).squeeze(1)  # (batch_size, hidden_size)

        # Add supplementary data
        pooled = pooled + supplementary_data

        # MLP
        y = self.mlp_fc1(pooled)
        y = self.mlp_activation(y)
        y = self.mlp_dropout(y)
        y = self.mlp_fc2(y)
        y = self.mlp_norm(pooled + y)  # residual connection

        # Project y to get m/z values and probabilities
        mzs = self.output_linear_mz(y)  # Shape: (batch_size, max_fragments)
        probs = self.output_linear_prob(y)  # Shape: (batch_size, max_fragments)

        # Apply activations
        mzs = self.mz_activation(mzs)
        probs = F.softmax(probs, dim=1)

        # Create a mask for probabilities above the threshold
        mask = probs > self.prob_threshold

        # Apply the mask to probabilities and renormalize
        probs = probs * mask.float()
        probs = probs / (probs.sum(dim=1, keepdim=True) + 1e-10)

        # Apply the mask to m/z values
        mzs = mzs * mask.float()

        # Stack together mzs and probs
        output = torch.stack([mzs, probs], dim=-1)  # Shape: (batch_size, max_fragments, 2)

        return output


class CustomChemBERTaModel(nn.Module):
    def __init__(self, model, max_fragments, max_seq_length, supplementary_data_dim, 
                 initial_sigma=1.0, final_sigma=1.0, eval_sigma=1.0, total_steps=1000000, 
                 prob_threshold=1e-4, mz_max_value=2000):
        super(CustomChemBERTaModel, self).__init__()
        self.model = model
        self.max_fragments = max_fragments
        self.max_seq_length = max_seq_length
        self.steps = 0
        self.hidden_size = self.model.config.hidden_size
        self.dim_supplementary_data = supplementary_data_dim
        self.mz_max_value = mz_max_value
        self.prob_threshold = prob_threshold
        self.final_layers = FinalLayers(
            hidden_size=self.hidden_size, 
            supplementary_data_dim=self.dim_supplementary_data, 
            max_fragments=self.max_fragments, 
            mz_max_value=self.mz_max_value,
            prob_threshold=self.prob_threshold
        )
        
        
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
        
    def extract_data(self, tensor):
        """
        Extract m/z values and probabilities/intensities from a tensor.
        
        :param tensor: Tensor of shape [batch_size, max_fragments, 2]
        :return: Tuple of (m/z values, probabilities/intensities)
        """
        return tensor[:, :, 0], tensor[:, :, 1]

    def calculate_loss(self, predictions, actual_labels, sigma):
        pred_mzs, pred_probabilities = self.extract_data(predictions)
        actual_mzs, actual_intensities = self.extract_data(actual_labels)
        actual_probabilities = actual_intensities / (torch.sum(actual_intensities, dim=1, keepdim=True) + 1e-8)

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
        Evaluate spectra using greedy cosine metric.
        
        :param predicted_output: Tensor of shape [batch_size, max_fragments, 2] (m/z, probabilities)
        :param labels: Ground truth labels Tensor of shape [batch_size, max_fragments, 2] (m/z, intensities)
        :return: A dictionary with the greedy cosine score.
        """
        pred_mz, pred_probs = self.extract_data(predicted_output)
        true_mz, true_intensities = self.extract_data(labels)
        
        greedy_scores = self.batch_greedy_cosine(pred_mz, true_mz, pred_probs, true_intensities)
        return {'greedy_score': greedy_scores.mean().item()}

    @staticmethod
    def batch_greedy_cosine(mz_a, mz_b, intensities_a, intensities_b, tolerance=0.1):
        """
        Compute greedy cosine similarity for a batch of spectra.
        
        :param mz_a: Tensor of shape [batch_size, max_fragments] - m/z values for spectra A
        :param mz_b: Tensor of shape [batch_size, max_fragments] - m/z values for spectra B
        :param intensities_a: Tensor of shape [batch_size, max_fragments] - intensities for spectra A
        :param intensities_b: Tensor of shape [batch_size, max_fragments] - intensities for spectra B
        :param tolerance: Float - tolerance for m/z matching
        :return: Tensor of shape [batch_size] - greedy cosine similarity scores
        """
        def create_spectrum(mz, intensities):
            # Convert to numpy and sort
            mz, intensities = map(lambda x: x.detach().cpu().numpy(), (mz, intensities))
            sorted_indices = np.argsort(mz)
            return Spectrum(mz=mz[sorted_indices], intensities=intensities[sorted_indices])

        cosine_greedy = CosineGreedy(tolerance=tolerance)
        
        scores = []
        for i in range(mz_a.shape[0]):
            spectrum_a = create_spectrum(mz_a[i], intensities_a[i])
            spectrum_b = create_spectrum(mz_b[i], intensities_b[i])
            
            try:
                result = cosine_greedy.pair(spectrum_a, spectrum_b)
                score, _ = result.tolist()
            except ZeroDivisionError:
                score = 1.0
            scores.append(score)

        return torch.tensor(scores, device=mz_a.device)



# ----- Testing -----

    
def test_model():
    print("Testing model")
    # Set up constants
    batch_size = 8
    max_fragments = 4
    max_seq_length = 10
    supplementary_data_dim = 81

    # Load the real ChemBERTa model and tokenizer
    chemBERTa_model = AutoModelForMaskedLM.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
    tokenizer = AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")

    # Initialize the CustomChemBERTaModel
    custom_model = CustomChemBERTaModel(chemBERTa_model, max_fragments, max_seq_length, supplementary_data_dim)

    # Create dummy input data
    input_smiles = ["CCONaCO", "CCC", "CCN", "CCCl", "CCBr", "CCCC", "CCCCO", "CCNCl"]
    input_encodings = tokenizer(
        input_smiles, 
        padding='max_length',
        truncation=True, 
        return_tensors="pt", 
        max_length=max_seq_length
    )

    input_ids = input_encodings["input_ids"]
    attention_mask = input_encodings["attention_mask"]
    supplementary_data = torch.randn(batch_size, supplementary_data_dim)

    # Generate random "labels" data
    mz_values = torch.randint(100, 401, (batch_size, max_fragments)).float()
    intensities = torch.rand((batch_size, max_fragments))
    probabilities = intensities / intensities.sum(dim=1, keepdim=True)
    labels = torch.stack([mz_values, probabilities], dim=-1)

    # Add noise to create "predicted" data
    sigma_mz = 1.0
    sigma_prob = 0.2
    noisy_mz = mz_values + torch.randn_like(mz_values) * sigma_mz
    noisy_prob = probabilities + torch.randn_like(probabilities) * sigma_prob
    noisy_prob = torch.clamp(noisy_prob, min=0)  # Ensure non-negative
    noisy_prob = noisy_prob / noisy_prob.sum(dim=1, keepdim=True)  # Renormalize
    predicted_output = torch.stack([noisy_mz, noisy_prob], dim=-1)

    # Calculate loss
    loss = custom_model.calculate_loss(predicted_output, labels, sigma=sigma_mz)
    print(f"Loss value: {loss.item()}")

    # Evaluate spectra
    evaluation_score = custom_model.evaluate_spectra(predicted_output, labels)
    print(f"Evaluation score: {evaluation_score}")

    # Test the forward pass
    with torch.no_grad():
        model_loss, model_output = custom_model(input_ids, attention_mask, supplementary_data, labels=labels)
        print(f"Model forward pass loss: {model_loss.item()}")
        print(f"Model output probs: {model_output[:, :, 1]}")
    
    peft_config = LoraConfig(
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["query", "value"],
        modules_to_save=["final_layers"], # change this to the name of the new modules at the end.
        bias="none"
    )

    peft_model = get_peft_model(custom_model, peft_config)


    model = get_peft_model(custom_model, peft_config)
    model.print_trainable_parameters()

if __name__ == "__main__":

    from peft import LoraConfig, get_peft_model
    test_model()

    