import numpy as np
from tests.base import FairnessBiasTest
from models.model_adapter import ModelAdapter

class PerformanceAcrossDemographicTest(FairnessBiasTest):
    """
    Evaluates model performance consistency across demographic groups,
    helping identify disparities in accuracy that could reflect bias.
    """
    def run(self, model_adapter: ModelAdapter, parameters: dict) -> dict:
        data = self.simulate_demographic_data(num_samples=parameters.get("num_samples", 100))
        results = {}
        for sample in data:
            group = sample["group"]
            # Get the model prediction for this sample.
            pred = model_adapter.get_prediction(sample["input"])
            pred_class = int(np.argmax(pred["prediction"]))
            correct = (pred_class == sample["label"])
            if group not in results:
                results[group] = {"correct": 0, "total": 0}
            results[group]["total"] += 1
            if correct:
                results[group]["correct"] += 1

        # Calculate accuracy per group safely.
        group_accuracy = {}
        for group, res in results.items():
            if res["total"] > 0:
                group_accuracy[group] = res["correct"] / res["total"]
            else:
                group_accuracy[group] = 0  # Safeguard: If no samples, set accuracy to 0

        overall_accuracy = np.mean(list(group_accuracy.values())) if group_accuracy else 0

        details = {"group_results": results}
        recommendations = []
        for group, acc in group_accuracy.items():
            if acc < parameters.get("min_accuracy", 0.7):
                recommendations.append(f"Low performance in {group} group: {acc:.2f}. Consider bias mitigation.")

        return self.format_test_result(
            passed=overall_accuracy >= parameters.get("min_overall_accuracy", 0.7),
            score=overall_accuracy,
            metrics={"group_accuracy": group_accuracy, "overall_accuracy": overall_accuracy},
            details=details,
            recommendations=recommendations
        )
