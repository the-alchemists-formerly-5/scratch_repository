import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForMaskedLM
from torch_geometric.nn import GraphNorm
from src.team5.models.gnn import GINEConv, ResBlock
from src.team5.models.graff import SignNet

class SMILESToGraphMapper:
    def __init__(self):
        # Placeholder for the mapping function
        pass

    def map_tokens_to_graph(self, smiles, tokens):
        # This function should be implemented to map SMILES tokens to atoms/bonds
        # For now, we'll return dummy data
        num_atoms = 10  # Example
        num_bonds = 9   # Example
        token_to_atom = {i: i % num_atoms for i in range(len(tokens))}
        token_to_bond = {i: i % num_bonds for i in range(len(tokens))}
        return token_to_atom, token_to_bond, num_atoms, num_bonds

class CombinedGrAFFChemBERTaModel(nn.Module):
    def __init__(self, chemBERTa_model, max_fragments, max_seq_length, supplementary_data_dim,
                 num_eigs=8, eig_dim=32, eig_depth=2, encoder_depth=6, decoder_depth=2,
                 dropout=0.1, initial_sigma=1.0, final_sigma=1.0, eval_sigma=1.0, 
                 total_steps=1000000, prob_threshold=1e-4, mz_max_value=2000):
        super(CombinedGrAFFChemBERTaModel, self).__init__()
        
        self.chemBERTa = chemBERTa_model
        self.hidden_size = self.chemBERTa.config.hidden_size
        self.max_fragments = max_fragments
        self.max_seq_length = max_seq_length
        self.supplementary_data_dim = supplementary_data_dim
        
        self.mapper = SMILESToGraphMapper()
        
        # GrAFF-MS components
        self.num_eigs = num_eigs
        self.eig_dim = eig_dim
        self.eig_depth = eig_depth
        self.encoder_depth = encoder_depth
        
        self.signnet = SignNet(
            num_eigs=num_eigs,
            embed_dim=self.hidden_size,
            rho_dim=self.hidden_size,
            rho_depth=eig_depth,
            phi_dim=eig_dim,
            phi_depth=eig_depth,
            dropout=dropout
        )
        
        self.encoder = nn.ModuleList([
            GINEConv(nn=ResBlock(self.hidden_size, dropout),
                     edge_dim=self.hidden_size,
                     eps=0,
                     train_eps=True)
            for _ in range(encoder_depth)
        ])
        
        self.edge_updates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(3 * self.hidden_size, self.hidden_size),
                nn.SiLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(self.hidden_size, self.hidden_size),
                GraphNorm(self.hidden_size)
            )
            for _ in range(encoder_depth)
        ])
        
        # Attention pooling
        self.attention_query = nn.Linear(self.hidden_size, 1)
        
        # Final layers (similar to your original FinalLayers)
        self.supplementary_linear = nn.Linear(supplementary_data_dim, self.hidden_size)
        self.supplementary_activation = nn.GELU()
        self.supplementary_dropout = nn.Dropout(dropout)
        self.supplementary_norm = nn.LayerNorm(self.hidden_size)
        
        self.mlp_fc1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.mlp_fc2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.mlp_activation = nn.GELU()
        self.mlp_dropout = nn.Dropout(dropout)
        self.mlp_norm = nn.LayerNorm(self.hidden_size)
        
        self.output_linear_mz = nn.Linear(self.hidden_size, self.max_fragments)
        self.output_linear_prob = nn.Linear(self.hidden_size, self.max_fragments)
        
        self.mz_activation = nn.ReLU()
        
        # Other parameters
        self.initial_sigma = initial_sigma
        self.final_sigma = final_sigma
        self.eval_sigma = eval_sigma
        self.total_steps = total_steps
        self.prob_threshold = prob_threshold
        self.mz_max_value = mz_max_value
        self.steps = 0
        self.training_mode = True

    def forward(self, input_ids, attention_mask, supplementary_data, smiles, eigenvecs, eigvals, edge_index, labels=None):
        # Get ChemBERTa embeddings
        chemBERTa_output = self.chemBERTa(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        token_embeddings = chemBERTa_output.hidden_states[-1]
        
        batch_size = input_ids.shape[0]
        
        # Process each molecule in the batch
        node_features = []
        edge_features = []
        batch_indices = []
        
        for i in range(batch_size):
            # Map tokens to graph
            token_to_atom, token_to_bond, num_atoms, num_bonds = self.mapper.map_tokens_to_graph(smiles[i], input_ids[i])
            
            # Initialize node and edge features
            node_feat = torch.zeros(num_atoms, self.hidden_size, device=input_ids.device)
            edge_feat = torch.zeros(num_bonds, self.hidden_size, device=input_ids.device)
            
            # Assign token embeddings to atoms and bonds
            for token, embedding in enumerate(token_embeddings[i]):
                if token in token_to_atom:
                    node_feat[token_to_atom[token]] += embedding
                if token in token_to_bond:
                    edge_feat[token_to_bond[token]] += embedding
            
            node_features.append(node_feat)
            edge_features.append(edge_feat)
            batch_indices.extend([i] * num_atoms)
        
        # Combine features from all molecules
        node_features = torch.cat(node_features, dim=0)
        edge_features = torch.cat(edge_features, dim=0)
        batch = torch.tensor(batch_indices, device=input_ids.device)
        
        # Apply GrAFF-MS algorithm
        x_eig = self.signnet(eigenvecs, eigvals[batch])
        x = node_features + x_eig
        
        for conv, edge_update in zip(self.encoder, self.edge_updates):
            x = conv(x, edge_index, edge_features)
            edge_features = edge_update(torch.cat([edge_features, x[edge_index[0]], x[edge_index[1]]], dim=1))
        
        # Attention pooling
        attention_scores = self.attention_query(x).squeeze(-1)
        attention_weights = F.softmax(attention_scores, dim=0)
        pooled = torch.scatter_add(x * attention_weights.unsqueeze(-1), batch, dim=0)
        
        # Process supplementary data
        supplementary_data = self.supplementary_linear(supplementary_data)
        supplementary_data = self.supplementary_dropout(supplementary_data)
        supplementary_data = self.supplementary_norm(supplementary_data)
        supplementary_data = self.supplementary_activation(supplementary_data)
        
        # Combine pooled graph features with supplementary data
        combined = pooled + supplementary_data
        
        # MLP
        y = self.mlp_fc1(combined)
        y = self.mlp_activation(y)
        y = self.mlp_dropout(y)
        y = self.mlp_fc2(y)
        y = self.mlp_norm(combined + y)  # residual connection
        
        # Output layers
        mzs = self.mz_activation(self.output_linear_mz(y))
        mzs = torch.clamp(mzs, max=self.mz_max_value)
        probs = F.softmax(self.output_linear_prob(y), dim=1)
        
        # Apply threshold and renormalize
        mask = probs > self.prob_threshold
        probs = probs * mask.float()
        probs = probs / (probs.sum(dim=1, keepdim=True) + 1e-10)
        mzs = mzs * mask.float()
        
        output = torch.stack([mzs, probs], dim=-1)
        
        if self.training_mode:
            sigma = self.initial_sigma - (self.initial_sigma - self.final_sigma) * (self.steps / self.total_steps)
            self.steps += 1
        else:
            sigma = self.eval_sigma
        
        if labels is not None:
            loss = self.calculate_loss(output, labels, sigma=sigma)
            return loss, output
        else:
            return output


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

    