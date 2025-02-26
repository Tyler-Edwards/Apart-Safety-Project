import streamlit as st
from typing import Dict, List, Any, Optional

# Import registry
from tests.registry import registry

# Import technical tests
from tests.technical.input_validation import InputValidationTest
from tests.technical.consistency import ConsistencyTest
from tests.technical.error_recovery import ErrorRecoveryTest
from tests.technical.load_test import LoadTest
from tests.technical.adversarial import TechnicalSafetyTest, AdvancedAdversarialTest

# Import fairness tests
from tests.fairness.demographic import PerformanceAcrossDemographicTest
from tests.fairness.disparate_impact import DisparateImpactEvaluationTest
from tests.fairness.bias_mitigation import BiasMitigationEffectivenessTest
from tests.fairness.intersectional import IntersectionalAnalysisTest

# Import compliance tests
from tests.compliance.standard import load_compliance_tests

# Import NLP-specific tests
from tests.nlp.robustness.linguistic_variation import LinguisticVariationTest
from tests.nlp.robustness.adversarial_attack import AdversarialAttackTest
from tests.nlp.robustness.instruction_following import InstructionFollowingTest
from tests.nlp.bias.linguistic_bias import LinguisticBiasTest
from tests.nlp.bias.generative_bias import GenerativeBiasTest, RepresentationBiasTest
from tests.nlp.safety.harmful_content import HarmfulContentTest
from tests.nlp.safety.text_generation_safety import (
    HarmfulContentGenerationTest, 
    SafetyGuardrailConsistencyTest, 
    PromptInjectionTest
)

# Import unified NLP tests
try:
    from tests.nlp.robustness.linguistic_variation import LinguisticVariationTest
    from tests.nlp.robustness.adversarial_attack import AdversarialAttackTest
    from tests.nlp.bias.linguistic_bias import LinguisticBiasTest
    from tests.nlp.safety.harmful_content import HarmfulContentTest
    UNIFIED_TESTS_AVAILABLE = True
except ImportError:
    UNIFIED_TESTS_AVAILABLE = False
    st.warning("Unified NLP tests not available. Some functionality may be limited.")

def register_all_tests():
    """
    Register all tests with the registry.
    """
    # Register technical tests
    registry.register_test(
        InputValidationTest(
            test_id="tech_safety_1",
            name="Input Validation Testing",
            description="Evaluates model behavior with various input types including edge cases"
        ),
        applicable_model_types=["all"],
        is_core=True,
        priority=1
    )
    
    registry.register_test(
        ConsistencyTest(
            test_id="tech_safety_2",
            name="Prediction Consistency",
            description="Tests model consistency across similar inputs"
        ),
        applicable_model_types=["all"],
        is_core=True,
        priority=1
    )
    
    registry.register_test(
        ErrorRecoveryTest(
            test_id="tech_safety_3",
            name="Error Recovery",
            description="Assesses how well the model handles and recovers from errors"
        ),
        applicable_model_types=["all"],
        is_core=True,
        priority=1
    )
    
    registry.register_test(
        LoadTest(
            test_id="tech_safety_4",
            name="Load Testing",
            description="Evaluates model performance under different load conditions"
        ),
        applicable_model_types=["all"],
        is_core=False,
        priority=2
    )
    
    registry.register_test(
        AdvancedAdversarialTest(
            test_id="tech_safety_5",
            name="Advanced Adversarial Testing",
            description="Evaluates model robustness under adversarial attacks"
        ),
        applicable_model_types=["all"],
        is_core=False,
        priority=2
    )
    
    # Register fairness tests
    registry.register_test(
        PerformanceAcrossDemographicTest(
            test_id="fairness_1",
            name="Performance Across Demographic Groups",
            description="Evaluate model performance across different demographic groups."
        ),
        applicable_model_types=["all"],
        is_core=True,
        priority=1
    )
    
    registry.register_test(
        DisparateImpactEvaluationTest(
            test_id="fairness_2",
            name="Disparate Impact Evaluation",
            description="Assess disparate impact by comparing positive outcome rates across groups."
        ),
        applicable_model_types=["all"],
        is_core=False,
        priority=2
    )
    
    registry.register_test(
        BiasMitigationEffectivenessTest(
            test_id="fairness_3",
            name="Bias Mitigation Effectiveness",
            description="Evaluate the effectiveness of bias mitigation strategies."
        ),
        applicable_model_types=["all"],
        is_core=False,
        priority=3
    )
    
    registry.register_test(
        IntersectionalAnalysisTest(
            test_id="fairness_4",
            name="Intersectional Analysis Engine",
            description="Analyze model performance across multiple demographic dimensions."
        ),
        applicable_model_types=["all"],
        is_core=False,
        priority=2
    )
    
    # Register compliance tests
    compliance_tests = load_compliance_tests()
    for i, test in enumerate(compliance_tests):
        registry.register_test(
            test,
            applicable_model_types=["all"],
            is_core=(i < 5),  # First 5 are core
            priority=2
        )
    
    # Register NLP-specific tests (note the specific model types)
    registry.register_test(
        LinguisticVariationTest(
            test_id="nlp_robustness_1",
            name="Linguistic Variation Testing",
            description="Evaluates model robustness to linguistic variations"
        ),
        applicable_model_types=["text"],
        is_core=False,
        priority=1
    )
    
    registry.register_test(
        AdversarialAttackTest(
            test_id="nlp_robustness_2",
            name="NLP Adversarial Attack Testing",
            description="Tests model robustness to adversarial text attacks"
        ),
        applicable_model_types=["text"],
        is_core=False,
        priority=2
    )
    
    registry.register_test(
        LinguisticBiasTest(
            test_id="nlp_bias_1",
            name="Linguistic Bias Evaluation",
            description="Detects linguistic biases across dimensions"
        ),
        applicable_model_types=["text"],
        is_core=False,
        priority=1
    )
    
    registry.register_test(
        HarmfulContentTest(
            test_id="nlp_safety_1",
            name="Harmful Content Detection",
            description="Tests model handling of harmful content"
        ),
        applicable_model_types=["text"],
        is_core=False,
        priority=1
    )
    
# Register unified NLP tests if available
    if UNIFIED_TESTS_AVAILABLE:
        # These are registered via decorator when imported
        pass


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
            description="Evaluates model robustness under adversarial attacks"
        )
    ]
    return technical_tests

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

def load_nlp_tests():
    """
    Load and return NLP-specific tests.
    """
    tests = []
    
    # NLP Robustness Tests
    tests.append(LinguisticVariationTest(
        test_id="nlp_robustness_1",
        name="Linguistic Variation Testing",
        description="Evaluates model robustness to linguistic variations"
    ))
    
    tests.append(AdversarialAttackTest(
        test_id="nlp_robustness_2",
        name="NLP Adversarial Attack Testing",
        description="Tests model robustness to adversarial text attacks"
    ))
    
    tests.append(InstructionFollowingTest(
        test_id="nlp_robustness_3",
        name="Instruction Following Testing",
        description="Tests model's ability to accurately follow complex instructions"
    ))
    
    # NLP Bias Tests
    tests.append(LinguisticBiasTest(
        test_id="nlp_bias_1",
        name="Linguistic Bias Evaluation",
        description="Detects linguistic biases across dimensions"
    ))
    
    tests.append(GenerativeBiasTest(
        test_id="nlp_bias_2",
        name="Generative Bias Evaluation",
        description="Tests for biases in text generation across demographic groups"
    ))
    
    tests.append(RepresentationBiasTest(
        test_id="nlp_bias_3",
        name="Representation Bias Testing",
        description="Tests for balanced representation in content generation"
    ))
    
    # NLP Safety Tests
    tests.append(HarmfulContentTest(
        test_id="nlp_safety_1",
        name="Harmful Content Detection",
        description="Tests model handling of harmful content"
    ))
    
    tests.append(HarmfulContentGenerationTest(
        test_id="nlp_safety_2",
        name="Harmful Content Generation Testing",
        description="Tests if a text generation model produces harmful content when prompted"
    ))
    
    tests.append(SafetyGuardrailConsistencyTest(
        test_id="nlp_safety_3",
        name="Safety Guardrail Consistency Testing",
        description="Tests consistency of safety guardrails against various prompt variations"
    ))
    
    tests.append(PromptInjectionTest(
        test_id="nlp_safety_4",
        name="Prompt Injection Resistance Testing",
        description="Tests resistance to prompt injection attacks"
    ))
    
    # Add unified tests if available
    if UNIFIED_TESTS_AVAILABLE:
        # Add unified tests with different IDs
        tests.append(LinguisticVariationTest(
            test_id="nlp_robustness_unified_1",
            name="Unified Linguistic Variation Testing",
            description="Unified test for linguistic variations"
        ))
        
        tests.append(AdversarialAttackTest(
            test_id="nlp_robustness_unified_2",
            name="Unified NLP Adversarial Attack Testing",
            description="Unified test for adversarial text attacks"
        ))
        
        tests.append(LinguisticBiasTest(
            test_id="nlp_bias_unified_1",
            name="Unified Linguistic Bias Evaluation",
            description="Unified test for linguistic biases"
        ))
        
        tests.append(HarmfulContentTest(
            test_id="nlp_safety_unified_1",
            name="Unified Harmful Content Detection",
            description="Unified test for harmful content"
        ))
    
    return tests

def load_test_catalog(model_type: Optional[str] = None):
    """
    Load the complete test catalog, filtered by model type if specified.
    
    Args:
        model_type: Optional model type to filter tests
        
    Returns:
        List: List of test instances
    """
    # Check if we should use the registry system
    use_registry = hasattr(registry, 'get_applicable_tests')
    
    if use_registry:
        # Ensure all tests are registered
        register_all_tests()
        
        # Get all tests or filter by model type
        if model_type:
            st.write(f"Loading tests for model type: {model_type}")
            test_ids = registry.get_applicable_tests(model_type)
        else:
            st.write("Loading all tests (no model type filter)")
            test_ids = registry.get_all_test_ids()
        
        # Group tests by category for logging
        tests_by_category = {}
        for test_id in test_ids:
            test = registry.get_test(test_id)
            if test is None:
                continue
            category = test.category
            if category not in tests_by_category:
                tests_by_category[category] = []
            tests_by_category[category].append(test)
        
        # Log test counts by category
        for category, tests in tests_by_category.items():
            st.write(f"Added {len(tests)} {category} tests")
        
        # Get the actual test instances
        tests = [registry.get_test(test_id) for test_id in test_ids if registry.get_test(test_id) is not None]
    else:
        # Use legacy approach
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
        
        # Only add NLP tests if appropriate model type
        if model_type in ["text", "multimodal"] or model_type is None:
            st.write("Loading NLP-Specific Tests...")
            nlp_tests = load_nlp_tests()
            st.write(f"Added {len(nlp_tests)} NLP-specific tests")
            tests.extend(nlp_tests)
        
        tests.extend(technical_tests)
        tests.extend(compliance_tests)
        tests.extend(fairness_tests)
    
    st.write(f"Total tests loaded: {len(tests)}")
    st.write("Test categories loaded:", sorted(set(test.category for test in tests)))
    
    return tests