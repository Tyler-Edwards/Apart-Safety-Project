import numpy as np
from tests.base import ComplianceTest

def load_compliance_tests():
    """
    Load and return a list of standard compliance tests.
    """
    tests = []
    categories = [
        "Regulatory Compliance",
        "Transparency",
        "Privacy Protection",
        "Operational Security"
    ]
    
    for category in categories:
        for i in range(5):
            test_id = f"{category.lower().replace(' ', '_')}_{i+1}"
            name = f"{category} Test {i+1}"
            description = f"Verifies {category.lower()} requirement #{i+1}"
            severity = np.random.choice(["low", "medium", "high", "critical"])
            tests.append(ComplianceTest(test_id, name, description, category, severity))
    
    return tests
