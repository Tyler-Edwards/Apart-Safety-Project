import streamlit as st
from catalog.test_catalog import load_test_catalog
from tests.technical.input_validation import InputValidationTest
from tests.technical.consistency import ConsistencyTest
from tests.technical.error_recovery import ErrorRecoveryTest
from tests.technical.load_test import LoadTest
from typing import Dict, Any

def configure_test_parameters(test):
    """
    Configure parameters for specific test types.
    This function handles the parameter UI for different types of technical safety tests.
    """
    parameters = {}
    
    if isinstance(test, InputValidationTest):
        parameters["min_success_rate"] = st.slider(
            "Minimum Success Rate",
            min_value=0.0,
            max_value=1.0,
            value=0.8,
            key=f"success_rate_{test.test_id}"
        )
        
    elif isinstance(test, ConsistencyTest):
        parameters["stability_threshold"] = st.slider(
            "Stability Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.8,
            key=f"stability_{test.test_id}"
        )
        parameters["consistency_threshold"] = st.slider(
            "Consistency Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            key=f"consistency_{test.test_id}"
        )
        
    elif isinstance(test, ErrorRecoveryTest):
        parameters["min_recovery_score"] = st.slider(
            "Minimum Recovery Score",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            key=f"recovery_{test.test_id}"
        )
        
    elif isinstance(test, LoadTest):
        parameters["min_performance_score"] = st.slider(
            "Minimum Performance Score",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            key=f"performance_{test.test_id}"
        )
    
    if parameters:
        st.session_state.test_parameters[test.test_id] = parameters

def render_test_config_page():
    """
    Renders the test configuration page where users can select
    and configure compliance tests.
    """
    st.title("Test Configuration")
    
    # Load and verify tests
    tests = load_test_catalog()
    st.write(f"Debug: Loaded {len(tests)} total tests")
    
    # Group tests by category and verify grouping
    tests_by_category = {}
    for test in tests:
        if test.category not in tests_by_category:
            tests_by_category[test.category] = []
        tests_by_category[test.category].append(test)
    
    # Debug: Print category information
    st.write("Debug: Categories found:", list(tests_by_category.keys()))
    for category, category_tests in tests_by_category.items():
        st.write(f"Debug: {category}: {len(category_tests)} tests")

    # Create tabs for categories
    tabs = st.tabs(list(tests_by_category.keys()))
    
    # Initialize test parameters if not exists
    if 'test_parameters' not in st.session_state:
        st.session_state.test_parameters = {}
    
    selected_tests = []
    
    # Display tests by category
    for tab, (category, category_tests) in zip(tabs, tests_by_category.items()):
        with tab:
            st.subheader(f"{category} Tests")
            
            # Select all option for category
            select_all = st.checkbox(
                f"Select all {category} tests",
                key=f"select_all_{category}"
            )
            
            for test in category_tests:
                with st.container():
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        # Test selection
                        selected = st.checkbox(
                            f"{test.name}",
                            value=select_all or test.test_id in st.session_state.selected_tests,
                            key=f"test_{test.test_id}",
                            help=test.description
                        )
                        
                        if selected:
                            selected_tests.append(test.test_id)
                            
                        # Show test description
                        st.caption(test.description)
                    
                    with col2:
                        # Configuration button for technical safety tests
                        if selected and isinstance(test, (InputValidationTest, ConsistencyTest, 
                                                        ErrorRecoveryTest, LoadTest)):
                            if st.button("Configure", key=f"config_{test.test_id}"):
                                st.session_state.configuring_test = test.test_id
                
                # Show configuration options if selected
                if (hasattr(st.session_state, 'configuring_test') and 
                    st.session_state.configuring_test == test.test_id):
                    
                    st.markdown("#### Test Parameters")
                    # Add appropriate configuration options based on test type
                    configure_test_parameters(test)
                
                st.markdown("---")  # Visual separator between tests
    
    # Save configuration
    if st.button("Save Test Configuration", key="save_test_config"):
        st.session_state.selected_tests = selected_tests
        st.success(f"Configuration saved with {len(selected_tests)} tests selected")
        
        if st.button("Proceed to Run Tests", key="proceed_to_run"):
            set_page("run_tests")

def set_page(page_name):
    """
    Changes the current page in the session state.
    """
    st.session_state.page = page_name