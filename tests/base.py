import time
import numpy as np
from datetime import datetime
from typing import Dict, List, Any
from models.model_adapter import ModelAdapter

class ComplianceTest:
    """
    Base class for all compliance tests.
    """
    def __init__(self, test_id, name, description, category, severity="medium"):
        self.test_id = test_id
        self.name = name
        self.description = description
        self.category = category
        self.severity = severity
        
    def run(self, model_adapter, parameters):
        """
        Run the compliance test with the given model and parameters.
        
        Args:
            model_adapter: The model adapter instance to test
            parameters: Dictionary of test parameters
            
        Returns:
            Dictionary containing test results
        """
        # Mock test execution
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


class BaseTechnicalSafetyTest(ComplianceTest):
    """
    Base class for technical safety tests that provides common functionality
    and standardized reporting methods.
    """
    def __init__(self, test_id: str, name: str, description: str):
        super().__init__(test_id, name, description, "Technical Safety")
        
    def format_test_result(self, passed: bool, score: float, metrics: Dict[str, Any], 
                          details: Dict[str, Any], recommendations: List[str]) -> Dict[str, Any]:
        """
        Standardizes the format of test results to ensure consistency across all technical
        safety tests while maintaining compatibility with the existing framework.
        """
        return {
            "pass": passed,
            "score": score,
            "message": f"Technical safety test completed with score={score:.2f}",
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics,
            "details": details,
            "recommendations": recommendations
        }

    def _generate_recommendations(self, results: List[Dict[str, Any]]) -> List[str]:
        """
        Generates generic recommendations based on test results.
        Can be overridden by subclasses for more specific recommendations.
        """
        recommendations = []
        
        failed_tests = [r for r in results if not r.get("handled", True)]
        
        if failed_tests:
            recommendations.append(
                "Consider improving input handling to avoid failures in specific cases."
            )

        if any("error" in r for r in results):
            recommendations.append(
                "Enhance error handling mechanisms to prevent unexpected failures."
            )

        if not recommendations:
            recommendations.append("No critical issues detected. Continue monitoring performance.")

        return recommendations


class FairnessBiasTest(ComplianceTest):
    """
    Base class for fairness and bias tests.
    """
    def __init__(self, test_id: str, name: str, description: str, category="Fairness & Bias"):
        super().__init__(test_id, name, description, category)
    
    def simulate_demographic_data(self, num_samples=100):
        """
        Generates synthetic test data representing different demographic groups.
        This version ensures that each group is represented at least once.
        """
        import random
        groups = ['Male', 'Female', 'Non-binary']
        data = []
        # First, add one sample for each group:
        for group in groups:
            input_text = f"Patient sample from {group} group"
            label = random.choice([0, 1])
            data.append({"group": group, "input": input_text, "label": label})
        # Then add additional random samples.
        for i in range(num_samples - len(groups)):
            group = random.choice(groups)
            input_text = f"Patient sample from {group} group"
            label = random.choice([0, 1])
            data.append({"group": group, "input": input_text, "label": label})
        return data

    def format_test_result(self, passed: bool, score: float, metrics: dict, details: dict, recommendations: list) -> dict:
        """
        Standardizes the format of test results.
        """
        from datetime import datetime
        return {
            "pass": passed,
            "score": score,
            "message": f"Test completed with score={score:.2f}",
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics,
            "details": details,
            "recommendations": recommendations
        }
