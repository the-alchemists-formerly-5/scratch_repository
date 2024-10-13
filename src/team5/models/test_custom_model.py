import torch
import pytest
from src.team5.models.custom_model import greedy_cosine_similarity_for_interleaved

@pytest.fixture
def batch_size():
    return 3

@pytest.fixture
def num_fragments():
    return 5

@pytest.fixture
def batched_input(batch_size, num_fragments):
    # Create batched predicted and actual vectors
    pred_vec = torch.rand(batch_size, num_fragments * 2) * 100
    actual_vec = torch.rand(batch_size, num_fragments * 2) * 600
    
    # Ensure the values are positive (as m/z and intensities should be)
    pred_vec = torch.abs(pred_vec)
    actual_vec = torch.abs(actual_vec)
    
    # Sort the vectors to simulate realistic data
    pred_vec_sorted, _ = torch.sort(pred_vec, dim=1, descending=True)
    actual_vec_sorted, _ = torch.sort(actual_vec, dim=1, descending=True)
    
    return pred_vec_sorted, actual_vec_sorted

def test_greedy_cosine_similarity_for_interleaved(batched_input):
    pred_vec, actual_vec = batched_input
    
    similarity = greedy_cosine_similarity_for_interleaved(pred_vec, actual_vec)

    print(similarity)
    
    # Check that the output has the correct shape
    assert similarity.shape == (pred_vec.shape[0],)
    
    # Check that all similarity values are between 0 and 1
    assert torch.all(similarity >= 0) and torch.all(similarity <= 1)
    
    # Test with identical inputs
    identical_similarity = greedy_cosine_similarity_for_interleaved(pred_vec, pred_vec)
    assert torch.allclose(identical_similarity, torch.ones_like(identical_similarity))
    
    # Test with completely different inputs
    diff_vec = 1 - actual_vec  # Invert the values
    different_similarity = greedy_cosine_similarity_for_interleaved(pred_vec, diff_vec)
    assert torch.all(different_similarity < identical_similarity)

if __name__ == "__main__":
    pytest.main(["-s", __file__])