import streamlit as st

# Import all test types
from tests.technical.input_validation import InputValidationTest
from tests.technical.consistency import ConsistencyTest
from tests.technical.error_recovery import ErrorRecoveryTest
from tests.technical.load_test import LoadTest
from tests.technical.adversarial import TechnicalSafetyTest, AdvancedAdversarialTest

from tests.fairness.demographic import PerformanceAcrossDemographicTest
from tests.fairness.disparate_impact import DisparateImpactEvaluationTest
from tests.fairness.bias_mitigation import BiasMitigationEffectivenessTest
from tests.fairness.intersectional import IntersectionalAnalysisTest

from tests.compliance.standard import load_compliance_tests

def load_fairness_tests():
    """
    Load and return a list of fairness & bias tests.
    """
    tests = []
    tests.append(PerformanceAcrossDemographicTest(
         test_id="fairness_1",
         name="Performance Across Demographic Groups",
         description="Evaluate model performance across different demographic groups."
    ))
    tests.append(DisparateImpactEvaluationTest(
         test_id="fairness_2",
         name="Disparate Impact Evaluation",
         description="Assess disparate impact by comparing positive outcome rates across groups."
    ))
    tests.append(BiasMitigationEffectivenessTest(
         test_id="fairness_3",
         name="Bias Mitigation Effectiveness",
         description="Evaluate the effectiveness of bias mitigation strategies."
    ))
    tests.append(IntersectionalAnalysisTest(
         test_id="fairness_4",
         name="Intersectional Analysis Engine",
         description="Analyze model performance across multiple demographic dimensions."
    ))
    return tests

def load_technical_tests():
    """
    Load and return a list of technical safety tests.
    """
    technical_tests = [
        InputValidationTest(
            test_id="tech_safety_1",
            name="Input Validation Testing",
            description="Evaluates model behavior with various input types including edge cases"
        ),
        ConsistencyTest(
            test_id="tech_safety_2",
            name="Prediction Consistency",
            description="Tests model consistency across similar inputs"
        ),
        ErrorRecoveryTest(
            test_id="tech_safety_3",
            name="Error Recovery",
            description="Assesses how well the model handles and recovers from errors"
        ),
        LoadTest(
            test_id="tech_safety_4",
            name="Load Testing",
            description="Evaluates model performance under different load conditions"
        ),
        AdvancedAdversarialTest(
            test_id="tech_safety_5",
            name="Advanced Adversarial Testing",
            description="Evaluates model robustness under adversarial attacks including sophisticated character-level modifications, synonym replacements, and TextAttack integration"
        )
    ]
    return technical_tests

def load_test_catalog():
    """
    Load the complete test catalog by combining technical safety tests,
    standard compliance tests, and fairness & bias tests.
    """
    tests = []
    
    st.write("Loading Technical Safety Tests...")
    technical_tests = load_technical_tests()
    st.write(f"Added {len(technical_tests)} technical safety tests")
    
    st.write("Loading Standard Compliance Tests...")
    compliance_tests = load_compliance_tests()
    st.write(f"Added {len(compliance_tests)} standard compliance tests")
    
    st.write("Loading Fairness & Bias Tests...")
    fairness_tests = load_fairness_tests()
    st.write(f"Added {len(fairness_tests)} fairness & bias tests")
    
    tests.extend(technical_tests)
    tests.extend(compliance_tests)
    tests.extend(fairness_tests)
    
    st.write(f"Total tests loaded: {len(tests)}")
    st.write("Test categories loaded:", sorted(set(test.category for test in tests)))
    
    return tests
