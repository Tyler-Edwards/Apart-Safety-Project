import streamlit as st
import pandas as pd
import numpy as np
import time
import plotly.express as px
import plotly.graph_objects as go

def get_compliance_color(score):
    """Return color based on compliance score"""
    if score >= 80:
        return "green"
    elif score >= 50:
        return "gold"
    else:
        return "red"

def render_results_page():
    """
    Renders the compliance test results page with multiple result views.
    """
    st.title("Compliance Test Results")
    
    if not st.session_state.test_results:
        st.warning("No test results available. Please run tests first.")
        st.button("Go to Run Tests", on_click=lambda: set_page("run_tests"))
        return
    
    # Tab selection for different views
    tab1, tab2, tab3, tab4 = st.tabs(["Summary", "Results by Category", "Detailed Results", "Export Report"])
    
    with tab1:
        render_results_summary()
    
    with tab2:
        render_results_by_category()
    
    with tab3:
        render_detailed_results()
    
    with tab4:
        render_export_options()

def render_results_summary():
    """
    Renders the summary dashboard with overall metrics and visualizations.
    """
    st.subheader("Summary Dashboard")
    
    # Calculate overall metrics
    test_results = st.session_state.test_results
    compliance_scores = st.session_state.compliance_scores
    
    total_tests = len(test_results)
    passed_tests = sum(1 for r in test_results.values() if r['result']['pass'])
    failed_tests = total_tests - passed_tests
    
    # Display overall metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Tests", total_tests)
    with col2:
        st.metric("Passed", passed_tests, f"+{passed_tests}")
    with col3:
        st.metric("Failed", failed_tests, f"-{failed_tests}", delta_color="inverse")
    
    # Calculate overall compliance score
    total_passed = sum(score["passed"] for score in compliance_scores.values())
    total_all = sum(score["total"] for score in compliance_scores.values())
    overall_score = (total_passed / total_all) * 100 if total_all > 0 else 0
    
    # Display overall compliance gauge
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = overall_score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Overall Compliance Score"},
        gauge = {
            'axis': {'range': [0, 100]},
            'bar': {'color': get_compliance_color(overall_score)},
            'steps': [
                {'range': [0, 50], 'color': "lightcoral"},
                {'range': [50, 80], 'color': "lightyellow"},
                {'range': [80, 100], 'color': "lightgreen"},
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 80
            }
        }
    ))
    
    fig.update_layout(height=250)
    st.plotly_chart(fig, use_container_width=True, key="overall_compliance_gauge")
    
    # Display category scores
    st.subheader("Compliance by Category")
    
    # Prepare data for horizontal bar chart
    categories = list(compliance_scores.keys())
    scores = [(score["passed"] / score["total"] * 100) if score["total"] > 0 else 0 
              for score in compliance_scores.values()]
    
    # Create horizontal bar chart
    fig = px.bar(
        x=scores, 
        y=categories, 
        orientation='h',
        labels={"x": "Compliance Score (%)", "y": "Category"},
        title="Compliance Score by Category",
        color=scores,
        color_continuous_scale=["red", "yellow", "green"],
        range_color=[0, 100]
    )
    
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True, key="category_bar_chart")

def render_results_by_category():
    """
    Renders detailed results organized by test category.
    """
    st.subheader("Results by Category")
    
    # Group results by category
    test_results = st.session_state.test_results
    results_by_category = {}
    
    for test_id, data in test_results.items():
        category = data["test"].category
        if category not in results_by_category:
            results_by_category[category] = []
        results_by_category[category].append(data)
    
    # Display results for each category
    for category, results in results_by_category.items():
        with st.expander(f"{category} ({sum(1 for r in results if r['result']['pass'])}/{len(results)} passed)"):
            # Display summary metrics for this category
            passed = sum(1 for r in results if r['result']['pass'])
            total = len(results)
            score = (passed / total) * 100 if total > 0 else 0
            
            st.metric(f"{category} Compliance Score", f"{score:.1f}%", 
                     delta=f"{passed}/{total} tests passed")
            
            # Display test results table
            results_data = []
            for data in results:
                test = data["test"]
                result = data["result"]
                results_data.append({
                    "Test ID": test.test_id,
                    "Test Name": test.name,
                    "Severity": test.severity.capitalize(),
                    "Status": "✅ Passed" if result["pass"] else "❌ Failed",
                    "Score": f"{result['score'] * 100:.1f}%"
                })
                
            df = pd.DataFrame(results_data)
            st.dataframe(df, hide_index=True, use_container_width=True)

def render_detailed_results():
    """
    Renders detailed test results with filtering options.
    """
    st.subheader("Detailed Test Results")
    
    test_results = st.session_state.test_results
    
    # Filter options
    col1, col2, col3 = st.columns(3)
    with col1:
        filter_status = st.multiselect(
            "Filter by Status",
            options=["Passed", "Failed"],
            default=["Passed", "Failed"]
        )
    with col2:
        filter_severity = st.multiselect(
            "Filter by Severity",
            options=["Low", "Medium", "High", "Critical"],
            default=["Low", "Medium", "High", "Critical"]
        )
    with col3:
        filter_category = st.multiselect(
            "Filter by Category",
            options=list(set(data["test"].category for data in test_results.values())),
            default=list(set(data["test"].category for data in test_results.values()))
        )
    
    # Apply filters
    filtered_results = {}
    for test_id, data in test_results.items():
        test = data["test"]
        result = data["result"]
        
        status = "Passed" if result["pass"] else "Failed"
        if (status in filter_status and 
            test.severity.capitalize() in filter_severity and
            test.category in filter_category):
            filtered_results[test_id] = data
    
    # Display results
    st.info(f"Showing {len(filtered_results)} of {len(test_results)} test results")
    
    if not filtered_results:
        st.warning("No results match the selected filters")
        return
    
    # Show test details in expandable sections
    for test_id, data in filtered_results.items():
        test = data["test"]
        result = data["result"]
        
        with st.expander(
            f"{test.name} ({test.category}) - {'✅ Passed' if result['pass'] else '❌ Failed'}",
            expanded=False
        ):
            cols = st.columns([3, 1])
            with cols[0]:
                st.markdown(f"**Description:** {test.description}")
                st.markdown(f"**Category:** {test.category}")
                st.markdown(f"**Severity:** {test.severity.capitalize()}")
            
            with cols[1]:
                gauge_fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=result["score"] * 100,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Test Score"},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': get_compliance_color(result["score"] * 100)},
                        'steps': [
                            {'range': [0, 50], 'color': "lightcoral"},
                            {'range': [50, 80], 'color': "lightyellow"},
                            {'range': [80, 100], 'color': "lightgreen"},
                        ]
                    }
                ))
                gauge_fig.update_layout(height=150, margin=dict(l=20, r=20, t=30, b=20))
                st.plotly_chart(gauge_fig, use_container_width=True, key=f"test_gauge_{test_id}")
            
            st.markdown("**Test Result Details:**")
            st.json(result)
            
            if not result["pass"]:
                st.markdown("**Remediation Suggestions:**")
                if "recommendations" in result and result["recommendations"]:
                    for rec in result["recommendations"]:
                        st.info(rec)
                else:
                    st.info("No specific remediation suggestions available.")

def render_export_options():
    """
    Renders the report export options panel.
    """
    st.subheader("Export Report")
    
    report_type = st.selectbox(
        "Report Type",
        options=["Full Compliance Report", "Executive Summary", "Technical Details", "Regulatory Evidence"]
    )
    
    format_options = ["PDF", "HTML", "JSON", "CSV"]
    report_format = st.selectbox("Format", options=format_options)
    
    include_options = st.multiselect(
        "Include in Report",
        options=["Test Details", "Remediation Suggestions", "Screenshots", "API Logs", "Configuration Settings"],
        default=["Test Details", "Remediation Suggestions"]
    )
    
    # Export button
    if st.button("Generate Report"):
        with st.spinner(f"Generating {report_type} in {report_format} format..."):
            time.sleep(2)  # Simulate report generation
            st.success(f"Report generated successfully!")
            
            # Display download link (mock)
            st.markdown(f"""
            <div style="text-align: center; padding: 10px; background-color: #f0f2f6; border-radius: 5px;">
                <p>Your report is ready for download:</p>
                <a href="#" style="text-decoration: none; padding: 10px; background-color: #4CAF50; color: white; border-radius: 5px;">
                    Download {report_type} ({report_format})
                </a>
            </div>
            """, unsafe_allow_html=True)

def set_page(page_name):
    """
    Changes the current page in the session state.
    """
    st.session_state.page = page_name