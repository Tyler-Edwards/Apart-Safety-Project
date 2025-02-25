import streamlit as st
import pandas as pd
import time
from catalog.test_catalog import load_test_catalog

def severity_level_to_num(severity):
    """
    Converts severity level string to numeric value for sorting/comparison.
    """
    return {"low": 1, "medium": 2, "high": 3, "critical": 4}.get(severity, 0)

def render_run_tests_page():
    """
    Renders the page where users can run selected compliance tests.
    """
    st.title("Run Compliance Tests")
    
    if not st.session_state.selected_tests:
        st.warning("No tests have been selected. Please go back to Test Configuration.")
        st.button("Go to Test Configuration", 
                 key="goto_test_config_button", 
                 on_click=lambda: set_page("test_config"))
        return
    
    # Display test summary
    st.subheader("Test Summary")
    
    tests = load_test_catalog()
    selected_test_objects = [test for test in tests if test.test_id in st.session_state.selected_tests]
    
    # Group by category
    tests_by_category = {}
    for test in selected_test_objects:
        if test.category not in tests_by_category:
            tests_by_category[test.category] = []
        tests_by_category[test.category].append(test)
    
    # Display summary
    summary_cols = st.columns(len(tests_by_category))
    for i, (category, cat_tests) in enumerate(tests_by_category.items()):
        with summary_cols[i]:
            st.metric(category, len(cat_tests))
    
    # Configure run options
    st.subheader("Run Configuration")
    
    with st.expander("Advanced Options", expanded=False):
        parallelism = st.slider("Test Parallelism", 
                              min_value=1, 
                              max_value=10, 
                              value=4,
                              key="parallelism_slider",
                              help="Number of tests to run simultaneously")
        
        timeout = st.number_input("Timeout per Test (seconds)", 
                                min_value=5, 
                                max_value=300, 
                                value=60,
                                key="timeout_input",
                                help="Maximum time allowed for each test to complete")
        
        verbose_logging = st.checkbox("Verbose Logging", 
                                    value=False,
                                    key="verbose_logging_checkbox",
                                    help="Show detailed test execution logs")
    
    # Run tests button
    if st.button("Run Tests", key="run_tests_button"):
        run_all_tests(selected_test_objects)

def run_all_tests(tests):
    """
    Executes all selected tests and stores results in session state.
    """
    # Initialize progress
    progress_bar = st.progress(0)
    status_text = st.empty()
    results_table = st.empty()
    current_results = []
    
    test_results = {}
    compliance_scores = {}
    
    # Run tests with simulated progress
    for i, test in enumerate(tests):
        # Update progress
        progress = (i / len(tests))
        progress_bar.progress(progress)
        status_text.text(f"Running test: {test.name} ({i+1}/{len(tests)})")
        
        # Run test
        parameters = st.session_state.test_parameters.get(test.test_id, {})  # Get configured parameters if available
        result = test.run(st.session_state.model_adapter, parameters)
        test_results[test.test_id] = {
            "test": test,
            "result": result
        }
        
        # Update category compliance scores
        if test.category not in compliance_scores:
            compliance_scores[test.category] = {"passed": 0, "total": 0}
        
        compliance_scores[test.category]["total"] += 1
        if result["pass"]:
            compliance_scores[test.category]["passed"] += 1
        
        # Add to current results
        current_results.append({
            "Test Name": test.name,
            "Category": test.category,
            "Status": "✅ Passed" if result["pass"] else "❌ Failed",
            "Score": f"{result['score'] * 100:.1f}%"
        })
        
        # Update results table
        df = pd.DataFrame(current_results)
        results_table.dataframe(df, hide_index=True, use_container_width=True)
    
    # Complete progress bar
    progress_bar.progress(1.0)
    status_text.text(f"Completed {len(tests)} tests")
    
    # Store results in session state
    st.session_state.test_results = test_results
    st.session_state.compliance_scores = compliance_scores
    
    # Show summary statistics
    st.subheader("Test Results Summary")
    
    # Calculate metrics
    total_passed = sum(1 for r in test_results.values() if r['result']['pass'])
    
    # Display metrics in columns
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Tests", len(tests))
    with col2:
        st.metric("Tests Passed", total_passed)
    with col3:
        st.metric("Tests Failed", len(tests) - total_passed)
    
    # Calculate and display overall compliance score
    total_score = sum(score["passed"] for score in compliance_scores.values())
    total_tests = sum(score["total"] for score in compliance_scores.values())
    overall_score = (total_score / total_tests) * 100 if total_tests > 0 else 0
    
    st.metric("Overall Compliance Score", f"{overall_score:.1f}%")
    
    # Display category breakdown
    st.subheader("Results by Category")
    category_data = []
    for category, scores in compliance_scores.items():
        category_score = (scores["passed"] / scores["total"]) * 100 if scores["total"] > 0 else 0
        category_data.append({
            "Category": category,
            "Tests Passed": scores["passed"],
            "Total Tests": scores["total"],
            "Compliance Score": f"{category_score:.1f}%"
        })
    
    category_df = pd.DataFrame(category_data)
    st.dataframe(category_df, hide_index=True, use_container_width=True)
    
    # Navigation to detailed results
    st.button("View Detailed Results", 
              key="view_results_button", 
              on_click=lambda: set_page("results"))

def set_page(page_name):
    """
    Changes the current page in the session state.
    """
    st.session_state.page = page_name
