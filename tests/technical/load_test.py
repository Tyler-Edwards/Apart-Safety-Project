import time
from typing import Dict, List, Any
from tests.base import BaseTechnicalSafetyTest
from models.model_adapter import ModelAdapter

class LoadTest(BaseTechnicalSafetyTest):
    """
    Tests the model's performance under various load conditions.
    """
    def __init__(self, test_id: str, name: str, description: str):
        super().__init__(test_id, name, description)
        
    def run(self, model_adapter: ModelAdapter, parameters: Dict[str, Any]) -> Dict[str, Any]:
        base_cases = [
            "Patient presents with fever and cough",
            "Individual reports severe headache",
            "Patient experiencing abdominal pain"
        ]
        
        # Test different load scenarios
        sequential_results = self._test_sequential_load(model_adapter, base_cases)
        parallel_results = self._test_parallel_load(model_adapter, base_cases)
        burst_results = self._test_burst_load(model_adapter, base_cases)
        
        # Calculate performance metrics
        metrics = {
            "sequential_success_rate": sequential_results["success_rate"],
            "parallel_success_rate": parallel_results["success_rate"],
            "burst_success_rate": burst_results["success_rate"],
            "average_response_time": (
                sequential_results["avg_time"] + 
                parallel_results["avg_time"] + 
                burst_results["avg_time"]
            ) / 3
        }
        
        overall_score = sum(metrics.values()) / len(metrics)
        
        return self.format_test_result(
            passed=overall_score >= parameters.get("min_performance_score", 0.7),
            score=overall_score,
            metrics=metrics,
            details={
                "sequential_tests": sequential_results,
                "parallel_tests": parallel_results,
                "burst_tests": burst_results
            },
            recommendations=self._generate_load_recommendations(metrics)
        )
    
    def _test_sequential_load(self, model_adapter, base_cases):
        success_count = 0
        times = []
        for case in base_cases:
            start = time.time()
            try:
                model_adapter.get_prediction(case)
                success_count += 1
            except Exception:
                pass
            times.append(time.time() - start)
        avg_time = sum(times) / len(times) if times else 0
        success_rate = success_count / len(base_cases) if base_cases else 1.0
        return {"success_rate": success_rate, "avg_time": avg_time}
    
    def _test_parallel_load(self, model_adapter, base_cases):
        # For a simple simulation, mimic parallel load with sequential calls
        success_count = 0
        times = []
        for case in base_cases:
            start = time.time()
            try:
                model_adapter.get_prediction(case)
                success_count += 1
            except Exception:
                pass
            times.append(time.time() - start)
        avg_time = sum(times) / len(times) if times else 0
        success_rate = success_count / len(base_cases) if base_cases else 1.0
        return {"success_rate": success_rate, "avg_time": avg_time}
    
    def _test_burst_load(self, model_adapter, base_cases):
        success_count = 0
        times = []
        burst_rounds = 3  # simulate burst load by repeating tests
        for _ in range(burst_rounds):
            for case in base_cases:
                start = time.time()
                try:
                    model_adapter.get_prediction(case)
                    success_count += 1
                except Exception:
                    pass
                times.append(time.time() - start)
        total_cases = len(base_cases) * burst_rounds if base_cases else 1
        avg_time = sum(times) / len(times) if times else 0
        success_rate = success_count / total_cases
        return {"success_rate": success_rate, "avg_time": avg_time}
    
    def _generate_load_recommendations(self, metrics: Dict[str, Any]) -> List[str]:
        recommendations = []
        if metrics["sequential_success_rate"] < 0.7:
            recommendations.append("Improve sequential processing performance.")
        if metrics["parallel_success_rate"] < 0.7:
            recommendations.append("Enhance parallel processing capability.")
        if metrics["burst_success_rate"] < 0.7:
            recommendations.append("Optimize burst load handling.")
        if not recommendations:
            recommendations.append("Load test performance is within acceptable limits.")
        return recommendations
