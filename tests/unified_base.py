"""
Unified base test classes that provide common functionality across all test types.
This eliminates duplication between general tests and specialized domain tests.
"""

import time
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Tuple
from models.model_adapter import ModelAdapter

class BaseComplianceTest:
    """
    Base class for all compliance tests with unified functionality.
    """
    def __init__(self, test_id: str, name: str, description: str, category: str, severity: str = "medium"):
        self.test_id = test_id
        self.name = name
        self.description = description
        self.category = category
        self.severity = severity
        
    def run(self, model_adapter: ModelAdapter, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the compliance test with the given model and parameters.
        
        Args:
            model_adapter: The model adapter instance to test
            parameters: Dictionary of test parameters
            
        Returns:
            Dictionary containing test results
        """
        # Mock test execution - should be overridden by subclasses
        time.sleep(np.random.uniform(0.5, 2.0))
        result = np.random.choice([True, False], p=[0.8, 0.2])
        details = {
            "pass": result,
            "score": np.random.uniform(0.5, 1.0) if result else np.random.uniform(0.1, 0.5),
            "message": f"Test {'passed' if result else 'failed'} with the given parameters",
            "timestamp": datetime.now().isoformat(),
            "parameters": parameters
        }
        return details
    
    def format_test_result(self, passed: bool, score: float, metrics: Dict[str, Any], 
                         details: Dict[str, Any], recommendations: List[str]) -> Dict[str, Any]:
        """
        Standardizes the format of test results for consistency across all tests.
        
        Args:
            passed: Whether the test passed
            score: Test score (0-1)
            metrics: Test metrics
            details: Test details
            recommendations: Recommendations based on test results
            
        Returns:
            Dict: Standardized test result
        """
        return {
            "pass": passed,
            "score": score,
            "message": f"Test completed with score={score:.2f}",
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics,
            "details": details,
            "recommendations": recommendations
        }
    
    def _generate_recommendations(self, results: List[Dict[str, Any]]) -> List[str]:
        """
        Generates generic recommendations based on test results.
        Can be overridden by subclasses for more specific recommendations.
        
        Args:
            results: Test results
            
        Returns:
            List[str]: Recommendations
        """
        recommendations = []
        
        # Default implementation checks for failures
        failed_results = [r for r in results if not r.get("pass", True)]
        
        if failed_results:
            recommendations.append(
                "Some tests failed. Consider reviewing the test details for more information."
            )
        else:
            recommendations.append("All tests passed. Continue monitoring performance.")

        return recommendations


class RobustnessTest(BaseComplianceTest):
    """
    Base class for all robustness tests (general and domain-specific).
    """
    def __init__(self, test_id: str, name: str, description: str, category: str = "Robustness"):
        super().__init__(test_id, name, description, category)
    
    def generate_variation(self, input_data: Any, variation_type: str, intensity: float = 0.1) -> Any:
        """
        Generate a variation of the input data based on the specified type.
        
        Args:
            input_data: Original input data
            variation_type: Type of variation
            intensity: Intensity of variation (0-1)
            
        Returns:
            Any: Varied input data
        """
        # Implementation depends on input type
        # Subclasses should override this for specific input types
        return input_data
    
    def compare_outputs(self, original_output: Any, varied_output: Any) -> float:
        """
        Compare original and varied outputs to determine robustness.
        
        Args:
            original_output: Output from original input
            varied_output: Output from varied input
            
        Returns:
            float: Similarity score (0-1)
        """
        # Basic implementation for general comparison
        # Domain-specific subclasses should override this
        if isinstance(original_output, dict) and isinstance(varied_output, dict):
            # Compare dictionary outputs
            return self._compare_dict_outputs(original_output, varied_output)
        
        elif isinstance(original_output, (list, np.ndarray)) and isinstance(varied_output, (list, np.ndarray)):
            # Compare list/array outputs
            return self._compare_array_outputs(original_output, varied_output)
        
        else:
            # Default comparison
            return 1.0 if original_output == varied_output else 0.0
    
    def _compare_dict_outputs(self, output1: Dict[str, Any], output2: Dict[str, Any]) -> float:
        """
        Compare two dictionary outputs and return similarity.
        
        Args:
            output1: First output
            output2: Second output
            
        Returns:
            float: Similarity score (0-1)
        """
        # Look for prediction or confidence values
        if "prediction" in output1 and "prediction" in output2:
            pred1 = np.array(output1["prediction"])
            pred2 = np.array(output2["prediction"])
            
            # L1 distance normalized to [0, 1]
            return 1.0 - float(np.sum(np.abs(pred1 - pred2)) / 2)
        
        elif "confidence" in output1 and "confidence" in output2:
            # Compare confidence scores
            conf1 = output1["confidence"]
            conf2 = output2["confidence"]
            return 1.0 - min(1.0, abs(conf1 - conf2))
        
        # Default moderate similarity
        return 0.5
    
    def _compare_array_outputs(self, output1: Union[List, np.ndarray], output2: Union[List, np.ndarray]) -> float:
        """
        Compare two array outputs and return similarity.
        
        Args:
            output1: First output
            output2: Second output
            
        Returns:
            float: Similarity score (0-1)
        """
        # Convert to numpy arrays
        arr1 = np.array(output1)
        arr2 = np.array(output2)
        
        # Check if shapes match
        if arr1.shape != arr2.shape:
            return 0.0
        
        # L1 distance normalized to [0, 1]
        max_diff = np.max(np.abs(arr1)) + np.max(np.abs(arr2))
        if max_diff == 0:
            return 1.0
        
        return 1.0 - float(np.sum(np.abs(arr1 - arr2)) / (max_diff * arr1.size))


class FairnessTest(BaseComplianceTest):
    """
    Base class for all fairness and bias tests (general and domain-specific).
    """
    def __init__(self, test_id: str, name: str, description: str, category: str = "Fairness & Bias"):
        super().__init__(test_id, name, description, category)
    
    def generate_test_pairs(self, template: Any, attribute1: Any, attribute2: Any) -> Tuple[Any, Any]:
        """
        Generate a pair of test cases that differ only in specified attributes.
        
        Args:
            template: Template for test cases
            attribute1: First attribute value
            attribute2: Second attribute value
            
        Returns:
            Tuple: Two test cases that should yield similar results if unbiased
        """
        # Basic implementation for string templates
        if isinstance(template, str) and "{ATTRIBUTE}" in template:
            test1 = template.replace("{ATTRIBUTE}", str(attribute1))
            test2 = template.replace("{ATTRIBUTE}", str(attribute2))
            return test1, test2
        
        # Default implementation
        return template, template
    
    def calculate_bias_score(self, output1: Any, output2: Any) -> float:
        """
        Calculate bias score between two outputs.
        
        Args:
            output1: Output for first test case
            output2: Output for second test case
            
        Returns:
            float: Bias score (0-1, where higher means more biased)
        """
        # Domain-specific subclasses should override this
        # Default implementation uses inverse of similarity
        similarity = self.compare_outputs(output1, output2)
        return 1.0 - similarity
    
    def compare_outputs(self, output1: Any, output2: Any) -> float:
        """
        Compare two outputs to determine similarity.
        
        Args:
            output1: First output
            output2: Second output
            
        Returns:
            float: Similarity score (0-1)
        """
        # Reuse the comparison logic from RobustnessTest
        if isinstance(output1, dict) and isinstance(output2, dict):
            return self._compare_dict_outputs(output1, output2)
        
        elif isinstance(output1, (list, np.ndarray)) and isinstance(output2, (list, np.ndarray)):
            return self._compare_array_outputs(output1, output2)
        
        else:
            return 1.0 if output1 == output2 else 0.0
    
    def _compare_dict_outputs(self, output1: Dict[str, Any], output2: Dict[str, Any]) -> float:
        """
        Compare two dictionary outputs and return similarity.
        
        Args:
            output1: First output
            output2: Second output
            
        Returns:
            float: Similarity score (0-1)
        """
        # Look for prediction or confidence values
        if "prediction" in output1 and "prediction" in output2:
            pred1 = np.array(output1["prediction"])
            pred2 = np.array(output2["prediction"])
            
            # L1 distance normalized to [0, 1]
            return 1.0 - float(np.sum(np.abs(pred1 - pred2)) / 2)
        
        elif "confidence" in output1 and "confidence" in output2:
            # Compare confidence scores
            conf1 = output1["confidence"]
            conf2 = output2["confidence"]
            return 1.0 - min(1.0, abs(conf1 - conf2))
        
        # Default moderate similarity
        return 0.5
    
    def _compare_array_outputs(self, output1: Union[List, np.ndarray], output2: Union[List, np.ndarray]) -> float:
        """
        Compare two array outputs and return similarity.
        
        Args:
            output1: First output
            output2: Second output
            
        Returns:
            float: Similarity score (0-1)
        """
        # Convert to numpy arrays
        arr1 = np.array(output1)
        arr2 = np.array(output2)
        
        # Check if shapes match
        if arr1.shape != arr2.shape:
            return 0.0
        
        # L1 distance normalized to [0, 1]
        max_diff = np.max(np.abs(arr1)) + np.max(np.abs(arr2))
        if max_diff == 0:
            return 1.0
        
        return 1.0 - float(np.sum(np.abs(arr1 - arr2)) / (max_diff * arr1.size))


class SafetyTest(BaseComplianceTest):
    """
    Base class for all safety and compliance tests (general and domain-specific).
    """
    def __init__(self, test_id: str, name: str, description: str, category: str = "Safety & Compliance"):
        super().__init__(test_id, name, description, category)
    
    def evaluate_safety(self, output: Any, criteria: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate output against safety criteria.
        
        Args:
            output: Model output to evaluate
            criteria: Safety criteria
            
        Returns:
            Dict: Safety evaluation results
        """
        # Domain-specific subclasses should override this
        return {"safe": True, "score": 1.0, "details": {}}
    
    def detect_sensitive_content(self, content: Any) -> List[Dict[str, Any]]:
        """
        Detect sensitive content in model input/output.
        
        Args:
            content: Content to analyze
            
        Returns:
            List: Detected sensitive content
        """
        # Domain-specific subclasses should override this
        return []