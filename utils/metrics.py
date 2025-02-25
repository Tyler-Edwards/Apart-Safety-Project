import numpy as np
import scipy.stats as stats

def compute_cosine_similarity(prob1, prob2):
    """
    Computes cosine similarity between two probability distributions.
    
    Args:
        prob1: First probability distribution
        prob2: Second probability distribution
        
    Returns:
        float: Cosine similarity score (0-1 range)
    """
    # Remove extra dimensions if needed
    prob1 = np.squeeze(prob1)
    prob2 = np.squeeze(prob2)
    
    dot_product = np.dot(prob1, prob2)
    norm_a = np.linalg.norm(prob1)
    norm_b = np.linalg.norm(prob2)
    return dot_product / (norm_a * norm_b)

def compute_kl_divergence(prob1, prob2):
    """
    Computes KL divergence between two probability distributions.
    
    Args:
        prob1: First probability distribution
        prob2: Second probability distribution
        
    Returns:
        float: KL divergence (higher values indicate greater difference)
    """
    epsilon = 1e-10
    p = np.clip(prob1, epsilon, 1)
    q = np.clip(prob2, epsilon, 1)
    return stats.entropy(p, q)
