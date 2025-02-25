import numpy as np
from tests.base import FairnessBiasTest
from models.model_adapter import ModelAdapter

class DisparateImpactEvaluationTest(FairnessBiasTest):
    """
    Evaluates disparate impact by comparing positive outcome rates across different demographic groups,
    using the standard four-fifths (80%) rule from employment law as a baseline.
    """
    def run(self, model_adapter: ModelAdapter, parameters: dict) -> dict:
        data = self.simulate_demographic_data(num_samples=parameters.get("num_samples", 100))
        group_positive = {}
        group_total = {}
        for sample in data:
            group = sample["group"]
            pred = model_adapter.get_prediction(sample["input"])
            pred_class = int(np.argmax(pred["prediction"]))
            if group not in group_total:
                group_total[group] = 0
                group_positive[group] = 0
            group_total[group] += 1
            if pred_class == 1:
                group_positive[group] += 1

        # Safely compute group positive rates.
        group_rates = {}
        for group in group_total:
            if group_total[group] > 0:
                group_rates[group] = group_positive[group] / group_total[group]
            else:
                group_rates[group] = 0

        # Use a baseline group (default "Male") for disparate impact comparison.
        baseline = parameters.get("baseline_group", "Male")
        baseline_rate = group_rates.get(baseline)
        # If baseline_rate is zero or missing, use a small epsilon to avoid division by zero.
        if baseline_rate is None or baseline_rate == 0:
            baseline_rate = 1e-10

        # Calculate disparate impact for each group relative to the baseline.
        disparate_impact = {}
        for group, rate in group_rates.items():
            disparate_impact[group] = rate / baseline_rate

        overall_ratio = np.mean(list(disparate_impact.values()))
        details = {
            "group_positive": group_positive,
            "group_total": group_total,
            "disparate_impact": disparate_impact
        }
        recommendations = []
        for group, ratio in disparate_impact.items():
            if ratio < parameters.get("min_ratio", 0.8) or ratio > parameters.get("max_ratio", 1.25):
                recommendations.append(
                    f"Disparate impact ratio for {group} is {ratio:.2f}. Consider reviewing fairness strategies."
                )
        passed = all(
            parameters.get("min_ratio", 0.8) <= ratio <= parameters.get("max_ratio", 1.25)
            for ratio in disparate_impact.values()
        )
        return self.format_test_result(
            passed=passed,
            score=overall_ratio,
            metrics={"disparate_impact": disparate_impact, "group_positive_rate": group_rates},
            details=details,
            recommendations=recommendations
        )
