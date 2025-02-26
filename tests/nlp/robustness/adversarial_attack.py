"""
Tests for NLP model robustness to adversarial attacks.
Uses the unified testing framework.
"""

import random
import numpy as np
from typing import Dict, List, Any, Tuple
from models.model_adapter import ModelAdapter
from tests.unified_base import RobustnessTest
from utils.perturbations import (
    generate_adversarial_examples,
    compare_predictions,
    homoglyph_substitution,
    word_insertion
)
from tests.registry import register

@register(
    applicable_model_types=["text", "multimodal"],
    is_core=False,
    priority=2
)
class AdversarialAttackTest(RobustnessTest):
    """
    Tests model robustness to adversarial text attacks:
    - Character-level perturbations
    - Word-level insertions
    - HotFlip attacks
    - Universal trigger attacks
    """
    
    def __init__(self, test_id: str, name: str, description: str):
        super().__init__(test_id, name, description, category="NLP Robustness")
    
    def run(self, model_adapter: ModelAdapter, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run adversarial attack tests.
        
        Args:
            model_adapter: Model adapter to test
            parameters: Test parameters
            
        Returns:
            Dict: Test results
        """
        # Get test parameters
        test_samples = parameters.get("test_samples", self._get_default_test_samples())
        attack_types = parameters.get("attack_types", ["character", "word", "mixed"])
        threshold = parameters.get("robustness_threshold", 0.7)
        num_examples = parameters.get("num_examples", 3)
        
        results = []
        
        # Run tests for each sample and attack type
        for sample in test_samples:
            sample_results = []
            original_prediction = model_adapter.get_prediction(sample)
            
            for attack_type in attack_types:
                # Generate adversarial examples using shared utility
                adversarial_examples = generate_adversarial_examples(
                    sample, attack_type, num_examples
                )
                
                attack_results = []
                for adv_example in adversarial_examples:
                    # Get prediction for adversarial example
                    adv_prediction = model_adapter.get_prediction(adv_example)
                    
                    # Compare predictions using shared utility
                    similarity = compare_predictions(original_prediction, adv_prediction)
                    
                    attack_results.append({
                        "adversarial_text": adv_example,
                        "similarity": similarity,
                        "prediction_changed": similarity < threshold
                    })
                
                # Calculate attack success rate (lower is better for model robustness)
                attack_success_rate = sum(1 for r in attack_results if r["prediction_changed"]) / len(attack_results)
                robustness_score = 1.0 - attack_success_rate
                
                sample_results.append({
                    "attack_type": attack_type,
                    "attack_results": attack_results,
                    "attack_success_rate": attack_success_rate,
                    "robustness_score": robustness_score
                })
            
            results.append({
                "original_text": sample,
                "original_prediction": original_prediction,
                "attack_results": sample_results
            })
        
        # Calculate overall robustness score
        avg_robustness = np.mean([
            result for sample in results
            for result in [r["robustness_score"] for r in sample["attack_results"]]
        ])
        
        # Generate recommendations
        recommendations = self._generate_recommendations(results)
        
        return self.format_test_result(
            passed=avg_robustness >= parameters.get("pass_threshold", 0.7),
            score=float(avg_robustness),
            metrics={
                "average_robustness": float(avg_robustness),
                "attack_types_tested": attack_types
            },
            details={"results": results},
            recommendations=recommendations
        )
    
    def _hotflip_attack(self, text: str) -> str:
        """
        Simplified implementation of HotFlip attack.
        
        In a real implementation, this would use gradients to find
        the most impactful character/word substitutions.
        
        Args:
            text: Original text
            
        Returns:
            str: Text with HotFlip-like modifications
        """
        # Simplified implementation - in real usage this would use model gradients
        # Here we just simulate it with targeted substitutions
        words = text.split()
        
        # Words that might flip sentiment/meaning
        sentiment_flippers = {
            "good": "great", "bad": "poor", "excellent": "decent",
            "terrible": "unfortunate", "increase": "rise", "decrease": "decline",
            "positive": "favorable", "negative": "unfavorable"
        }
        
        for i, word in enumerate(words):
            lower_word = word.lower()
            if lower_word in sentiment_flippers:
                words[i] = sentiment_flippers[lower_word]
                break
        
        return " ".join(words)
    
    def _get_default_test_samples(self) -> List[str]:
        """
        Get default test samples if none are provided.
        
        Returns:
            List[str]: Default test samples
        """
        return [
            "This product is excellent and I highly recommend it.",
            "The service was terrible and I would not recommend it to anyone.",
            "The movie was neither good nor bad, just average.",
            "The company reported a significant increase in quarterly profits.",
            "The patient's condition has improved following the treatment."
        ]
    
    def _generate_recommendations(self, results: List[Dict[str, Any]]) -> List[str]:
        """
        Generate recommendations based on test results.
        
        Args:
            results: Test results
            
        Returns:
            List[str]: Recommendations
        """
        recommendations = []
        
        # Analyze results by attack type
        attack_type_scores = {}
        for sample in results:
            for attack_result in sample["attack_results"]:
                attack_type = attack_result["attack_type"]
                score = attack_result["robustness_score"]
                
                if attack_type not in attack_type_scores:
                    attack_type_scores[attack_type] = []
                
                attack_type_scores[attack_type].append(score)
        
        # Generate attack-specific recommendations
        for attack_type, scores in attack_type_scores.items():
            avg_score = np.mean(scores)
            
            if avg_score < 0.6:
                if attack_type == "character":
                    recommendations.append(
                        "Model shows high vulnerability to character-level attacks. Consider adding character-level "
                        "perturbations during training or implementing character-aware models."
                    )
                elif attack_type == "word":
                    recommendations.append(
                        "Model is vulnerable to word insertion attacks. Consider training with augmented data that "
                        "includes benign word insertions."
                    )
                elif attack_type == "mixed":
                    recommendations.append(
                        "Model shows vulnerability to mixed attacks. Consider adversarial "
                        "training with various types of textual perturbations."
                    )
            elif avg_score < 0.8:
                recommendations.append(
                    f"Model shows moderate robustness to {attack_type} attacks. "
                    f"Consider additional adversarial training."
                )
        
        if not recommendations:
            recommendations.append(
                "Model demonstrates good robustness to tested adversarial attacks. "
                "Consider testing with more advanced attack methods."
            )
        
        return recommendations