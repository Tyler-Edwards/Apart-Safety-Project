"""
Tests for NLP model robustness to linguistic variations and transformations.
Uses the unified testing framework.
"""

import random
from typing import Dict, List, Any
import numpy as np
from models.model_adapter import ModelAdapter
from tests.unified_base import RobustnessTest
from utils.perturbations import (
    character_swap, 
    character_perturbation, 
    word_insertion,
    case_transformation,
    homoglyph_substitution,
    generate_adversarial_examples,
    compare_predictions
)
from tests.registry import register

@register(
    applicable_model_types=["text", "multimodal"],
    is_core=False,
    priority=1
)
class LinguisticVariationTest(RobustnessTest):
    """
    Tests model robustness to various linguistic variations:
    - Spelling errors and typos
    - Synonym replacements
    - Word order changes
    - Passive/active voice transformations
    - Contractions/expansions
    """
    
    def __init__(self, test_id: str, name: str, description: str):
        super().__init__(test_id, name, description, category="NLP Robustness")
    
    def run(self, model_adapter: ModelAdapter, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the linguistic variation test.
        
        Args:
            model_adapter: The model adapter to test
            parameters: Test parameters
            
        Returns:
            Dict: Test results
        """
        # Get test parameters
        test_samples = parameters.get("test_samples", self._get_default_test_samples())
        variation_intensity = parameters.get("variation_intensity", 0.2)
        variation_types = parameters.get("variation_types", [
            "character_swap", 
            "character_perturbation", 
            "word_insertion",
            "case_transformation"
        ])
        threshold = parameters.get("similarity_threshold", 0.7)
        
        results = []
        
        # Test each sample with different variation types
        for sample in test_samples:
            sample_results = []
            original_prediction = model_adapter.get_prediction(sample)
            
            for variation_type in variation_types:
                # Generate variation using shared utilities
                varied_text = self.generate_variation(sample, variation_type, variation_intensity)
                
                # Get prediction for varied text
                varied_prediction = model_adapter.get_prediction(varied_text)
                
                # Compare predictions using shared utility
                similarity = compare_predictions(original_prediction, varied_prediction)
                
                sample_results.append({
                    "original_text": sample,
                    "varied_text": varied_text,
                    "variation_type": variation_type,
                    "similarity": similarity,
                    "passed": similarity >= threshold
                })
            
            results.append({
                "sample": sample,
                "variations": sample_results
            })
        
        # Calculate overall score
        passed_variations = sum(1 for r in results for v in r["variations"] if v["passed"])
        total_variations = sum(len(r["variations"]) for r in results)
        overall_score = passed_variations / total_variations if total_variations > 0 else 0
        
        # Generate recommendations
        recommendations = self._generate_recommendations(results)
        
        return self.format_test_result(
            passed=overall_score >= parameters.get("pass_threshold", 0.8),
            score=overall_score,
            metrics={
                "robustness_score": overall_score,
                "passed_variations": passed_variations,
                "total_variations": total_variations
            },
            details={"results": results},
            recommendations=recommendations
        )
    
    def generate_variation(self, text: str, variation_type: str, intensity: float = 0.1) -> str:
        """
        Generate variation using the appropriate utility function.
        
        Args:
            text: Input text
            variation_type: Type of variation
            intensity: Intensity of variation
            
        Returns:
            str: Varied text
        """
        if variation_type == "character_swap":
            return character_swap(text, intensity)
        elif variation_type == "character_perturbation":
            return character_perturbation(text, intensity)
        elif variation_type == "word_insertion":
            return word_insertion(text, intensity)
        elif variation_type == "case_transformation":
            return case_transformation(text, intensity)
        elif variation_type == "homoglyph":
            return homoglyph_substitution(text, intensity)
        else:
            # Default to character swap if unknown type
            return character_swap(text, intensity)
    
    def _get_default_test_samples(self) -> List[str]:
        """
        Get default test samples if none are provided.
        
        Returns:
            List[str]: Default test samples
        """
        return [
            "The company announced a new product launch next quarter.",
            "The patient showed symptoms of fever and fatigue.",
            "The financial report indicates a 15% increase in revenue.",
            "The algorithm correctly classified 92% of the test cases.",
            "The researchers published their findings in a peer-reviewed journal."
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
        
        # Analyze results by variation type
        variation_results = {}
        for result in results:
            for variation in result["variations"]:
                var_type = variation["variation_type"]
                if var_type not in variation_results:
                    variation_results[var_type] = {"passed": 0, "total": 0}
                
                variation_results[var_type]["total"] += 1
                if variation["passed"]:
                    variation_results[var_type]["passed"] += 1
        
        # Generate specific recommendations
        for var_type, counts in variation_results.items():
            score = counts["passed"] / counts["total"] if counts["total"] > 0 else 0
            
            if score < 0.7:
                if var_type == "character_swap" or var_type == "character_perturbation":
                    recommendations.append(
                        "Model shows low robustness to character-level changes. Consider data augmentation with misspelled examples."
                    )
                elif var_type == "word_insertion":
                    recommendations.append(
                        "Model is sensitive to word insertions. Consider training with varied sentence structures."
                    )
                elif var_type == "case_transformation":
                    recommendations.append(
                        "Model is sensitive to case changes. Consider case-insensitive preprocessing or augmentation."
                    )
        
        if not recommendations:
            recommendations.append("Model shows good robustness to linguistic variations.")
        
        return recommendations