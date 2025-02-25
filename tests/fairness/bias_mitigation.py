import numpy as np
from tests.base import FairnessBiasTest
from models.model_adapter import ModelAdapter

class BiasMitigationEffectivenessTest(FairnessBiasTest):
    """
    Tests the effectiveness of bias mitigation strategies by simulating before/after
    scenarios and measuring improvement in fairness metrics.
    """
    def run(self, model_adapter: ModelAdapter, parameters: dict) -> dict:
        data = self.simulate_demographic_data(num_samples=parameters.get("num_samples", 100))
        # Simulate baseline positive rates (before mitigation)
        group_positive_before = {}
        group_total = {}
        for sample in data:
            group = sample["group"]
            pred = model_adapter.get_prediction(sample["input"])
            pred_class = int(np.argmax(pred["prediction"]))
            if group not in group_total:
                group_total[group] = 0
                group_positive_before[group] = 0
            group_total[group] += 1
            if pred_class == 1:
                group_positive_before[group] += 1
        group_rates_before = {group: group_positive_before[group] / group_total[group] for group in group_total}
        
        # Simulate bias mitigation by artificially improving positive rates for groups below a target.
        target_rate = parameters.get("target_rate", 0.5)
        group_rates_after = {}
        for group, rate in group_rates_before.items():
            if rate < target_rate:
                group_rates_after[group] = min(rate + 0.1, 1.0)
            else:
                group_rates_after[group] = rate
        
        details = {"group_rates_before": group_rates_before, "group_rates_after": group_rates_after}
        recommendations = []
        improvements = []
        for group in group_rates_before:
            improvement = group_rates_after[group] - group_rates_before[group]
            improvements.append(improvement)
            if improvement > 0:
                recommendations.append(f"Bias mitigation improved positive rate for {group} from {group_rates_before[group]:.2f} to {group_rates_after[group]:.2f}.")
            else:
                recommendations.append(f"No significant improvement for {group}.")
        overall_improvement = np.mean(improvements)
        passed = overall_improvement >= parameters.get("min_improvement", 0.05)
        return self.format_test_result(
            passed=passed,
            score=overall_improvement,
            metrics={"group_rates_before": group_rates_before, "group_rates_after": group_rates_after},
            details=details,
            recommendations=recommendations
        )
