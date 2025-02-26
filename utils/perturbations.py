"""
Shared utility functions for test input perturbations.
Provides reusable components for robustness testing across model types.
"""

import random
import string
import numpy as np
from typing import List, Dict, Any, Union, Tuple

def character_swap(text: str, intensity: float = 0.1) -> str:
    """
    Swap adjacent characters in text with given intensity.
    
    Args:
        text: Input text
        intensity: Proportion of word characters to swap (0-1)
        
    Returns:
        str: Text with swapped characters
    """
    words = text.split()
    num_words_to_modify = max(1, int(len(words) * intensity))
    indices = random.sample(range(len(words)), min(num_words_to_modify, len(words)))
    
    for idx in indices:
        word = words[idx]
        if len(word) <= 1:
            continue
            
        # Swap two adjacent characters
        char_idx = random.randint(0, len(word) - 2)
        word = word[:char_idx] + word[char_idx+1] + word[char_idx] + word[char_idx+2:]
        words[idx] = word
    
    return " ".join(words)

def character_perturbation(text: str, intensity: float = 0.1) -> str:
    """
    Apply random character perturbations: insertion, deletion, substitution.
    
    Args:
        text: Input text
        intensity: Proportion of characters to modify
        
    Returns:
        str: Text with perturbed characters
    """
    words = text.split()
    num_to_modify = max(1, int(len(words) * intensity))
    indices = random.sample(range(len(words)), min(num_to_modify, len(words)))
    
    for idx in indices:
        word = words[idx]
        if len(word) <= 1:
            continue
            
        perturbation_type = random.choice(["replace", "insert", "delete"])
        
        if perturbation_type == "replace" and len(word) >= 1:
            # Replace a character with a random one
            char_idx = random.randint(0, len(word) - 1)
            random_char = random.choice(string.ascii_lowercase)
            word = word[:char_idx] + random_char + word[char_idx+1:]
            
        elif perturbation_type == "insert":
            # Insert a random character
            char_idx = random.randint(0, len(word))
            random_char = random.choice(string.ascii_lowercase)
            word = word[:char_idx] + random_char + word[char_idx:]
            
        elif perturbation_type == "delete" and len(word) >= 2:
            # Delete a character
            char_idx = random.randint(0, len(word) - 1)
            word = word[:char_idx] + word[char_idx+1:]
        
        words[idx] = word
    
    return " ".join(words)

def homoglyph_substitution(text: str, intensity: float = 0.1) -> str:
    """
    Replace characters with visually similar ones (homoglyphs).
    
    Args:
        text: Input text
        intensity: Proportion of eligible characters to replace
        
    Returns:
        str: Text with homoglyph substitutions
    """
    homoglyphs = {
        'a': ['а', 'ɑ'], 'e': ['е', 'ē'], 'i': ['і', 'ı'], 
        'o': ['о', 'ο'], 's': ['ѕ'], 'c': ['с'], 'p': ['р'],
        'x': ['х'], 'y': ['у']
    }
    
    result = ""
    replaced_count = 0
    eligible_chars = sum(1 for c in text.lower() if c in homoglyphs)
    target_replacements = max(1, int(eligible_chars * intensity))
    
    for char in text:
        if char.lower() in homoglyphs and replaced_count < target_replacements and random.random() < intensity:
            replacement = random.choice(homoglyphs[char.lower()])
            result += replacement
            replaced_count += 1
        else:
            result += char
    
    return result

def word_insertion(text: str, intensity: float = 0.1) -> str:
    """
    Insert innocuous words into text.
    
    Args:
        text: Input text
        intensity: Controls number of insertions
        
    Returns:
        str: Text with inserted words
    """
    words = text.split()
    
    # Words that might not significantly change meaning
    benign_insertions = [
        "basically", "actually", "literally", "definitely", "really",
        "very", "quite", "just", "simply", "perhaps", "maybe"
    ]
    
    # Number of words to insert
    num_insertions = max(1, int(len(words) * intensity))
    
    for _ in range(num_insertions):
        insert_pos = random.randint(0, len(words))
        insert_word = random.choice(benign_insertions)
        words.insert(insert_pos, insert_word)
    
    return " ".join(words)

def case_transformation(text: str, intensity: float = 0.1) -> str:
    """
    Transform case of random words (uppercase, lowercase, title case).
    
    Args:
        text: Input text
        intensity: Proportion of words to transform
        
    Returns:
        str: Text with case transformations
    """
    words = text.split()
    num_to_modify = max(1, int(len(words) * intensity))
    indices = random.sample(range(len(words)), min(num_to_modify, len(words)))
    
    for idx in indices:
        word = words[idx]
        if not word:
            continue
            
        transform_type = random.choice(["upper", "lower", "title"])
        
        if transform_type == "upper":
            words[idx] = word.upper()
        elif transform_type == "lower":
            words[idx] = word.lower()
        elif transform_type == "title":
            words[idx] = word.title()
    
    return " ".join(words)

def generate_numeric_perturbation(value: Union[int, float], intensity: float = 0.1) -> Union[int, float]:
    """
    Perturb a numeric value by a random factor.
    
    Args:
        value: Numeric value to perturb
        intensity: Controls the magnitude of perturbation
        
    Returns:
        Union[int, float]: Perturbed value
    """
    # Apply random perturbation within intensity bounds
    perturbation = random.uniform(-intensity, intensity)
    perturbed_value = value * (1 + perturbation)
    
    # Return int if original was int
    if isinstance(value, int):
        return int(round(perturbed_value))
    return perturbed_value

def generate_image_perturbation(image: np.ndarray, intensity: float = 0.1) -> np.ndarray:
    """
    Apply perturbations to image data.
    
    Args:
        image: Image array (numpy)
        intensity: Controls the magnitude of perturbation
        
    Returns:
        np.ndarray: Perturbed image
    """
    # Add random noise
    noise = np.random.normal(0, intensity * 255, image.shape)
    perturbed_image = image + noise
    
    # Ensure valid pixel range
    perturbed_image = np.clip(perturbed_image, 0, 255)
    
    return perturbed_image.astype(image.dtype)

def generate_adversarial_examples(text: str, attack_type: str, num_examples: int = 3) -> List[str]:
    """
    Generate adversarial examples using various techniques.
    
    Args:
        text: Original text
        attack_type: Type of attack ("character", "word", "mixed")
        num_examples: Number of examples to generate
        
    Returns:
        List[str]: List of adversarial examples
    """
    examples = []
    
    for _ in range(num_examples):
        if attack_type == "character":
            # Character-level attacks
            attack_subtype = random.choice(["swap", "perturbation", "homoglyph"])
            if attack_subtype == "swap":
                examples.append(character_swap(text, random.uniform(0.1, 0.3)))
            elif attack_subtype == "perturbation":
                examples.append(character_perturbation(text, random.uniform(0.1, 0.3)))
            elif attack_subtype == "homoglyph":
                examples.append(homoglyph_substitution(text, random.uniform(0.1, 0.2)))
                
        elif attack_type == "word":
            # Word-level attacks
            examples.append(word_insertion(text, random.uniform(0.1, 0.3)))
            
        elif attack_type == "mixed":
            # Combine multiple attack types
            perturbed = text
            perturbed = character_perturbation(perturbed, random.uniform(0.05, 0.1))
            perturbed = word_insertion(perturbed, random.uniform(0.05, 0.1))
            examples.append(perturbed)
            
        else:
            # Default to character swap
            examples.append(character_swap(text, 0.2))
    
    return examples

def compare_predictions(pred1: Dict[str, Any], pred2: Dict[str, Any]) -> float:
    """
    Compare two model predictions and return a similarity score.
    
    Args:
        pred1: First prediction
        pred2: Second prediction
        
    Returns:
        float: Similarity score (0-1)
    """
    # Check for prediction arrays
    if "prediction" in pred1 and "prediction" in pred2:
        p1 = np.array(pred1["prediction"])
        p2 = np.array(pred2["prediction"])
        
        # For classification models, compare probability distributions
        if len(p1.shape) > 1 and p1.shape[1] > 1:
            # Calculate similarity (1 - L1 distance)
            return 1.0 - float(np.sum(np.abs(p1 - p2)) / 2)
        
        # For scalar outputs, compute normalized similarity
        return 1.0 - min(1.0, float(np.abs(p1 - p2)) / max(np.abs(p1), 1e-10))
    
    # Compare confidence scores if available
    if "confidence" in pred1 and "confidence" in pred2:
        return 1.0 - min(1.0, abs(pred1["confidence"] - pred2["confidence"]))
    
    # Default fallback
    return 0.5