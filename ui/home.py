import streamlit as st

def render_home_page():
    """
    Renders the home page of the AI Compliance Testing Platform.
    """
    st.title("AI Compliance Testing Platform")
    
    st.markdown("""
    ### Welcome to the AI Compliance Testing Platform
    
    Ensure your AI models meet regulatory requirements and ethical standards with our comprehensive testing suite.
    
    #### How it works:
    1. Configure your model details
    2. Select relevant compliance tests
    3. Run tests and review results
    4. Generate compliance reports
    
    Get started by configuring your model using the "Model Configuration" button in the sidebar.
    """)
    
    # Display compliance framework coverage
    st.subheader("Supported Compliance Frameworks")
    
    frameworks = {
        "GDPR": "European data protection regulation",
        "CCPA": "California Consumer Privacy Act",
        "EU AI Act": "Proposed European regulation for AI systems",
        "NIST AI RMF": "NIST AI Risk Management Framework",
        "ISO/IEC 42001": "AI management system standard",
        "HIPAA": "Health Insurance Portability and Accountability Act",
    }
    
    cols = st.columns(3)
    for i, (framework, description) in enumerate(frameworks.items()):
        with cols[i % 3]:
            st.markdown(f"**{framework}**")
            st.caption(description)
    
    # Sample stats
    st.subheader("Platform Statistics")
    metrics_cols = st.columns(4)
    metrics_cols[0].metric("Total Tests Available", "400+")
    metrics_cols[1].metric("Supported Model Types", "50")
    metrics_cols[2].metric("Compliance Frameworks", "12")
    metrics_cols[3].metric("Avg. Testing Time", "8 min")
