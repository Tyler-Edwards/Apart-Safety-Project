import streamlit as st
import nltk

# Import UI components
from ui.home import render_home_page
from ui.model_config import render_model_config_page
from ui.test_config import render_test_config_page
from ui.run_tests import render_run_tests_page, run_all_tests
from ui.results import render_results_page

# Import catalog
from catalog.test_catalog import load_test_catalog

# Ensure NLTK dependencies are downloaded
nltk.download('punkt_tab', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

def set_page(page_name):
    """
    Helper function to change pages
    """
    st.session_state.page = page_name

def main():
    """
    Main function that sets up the Streamlit app and handles navigation
    """
    st.set_page_config(page_title="AI Compliance Testing Tool", layout="wide")
    
    # Initialize session state for storing app state
    if 'page' not in st.session_state:
        st.session_state.page = "home"
    if 'model_configured' not in st.session_state:
        st.session_state.model_configured = False
    if 'test_results' not in st.session_state:
        st.session_state.test_results = {}
    if 'selected_tests' not in st.session_state:
        st.session_state.selected_tests = []
    if 'compliance_scores' not in st.session_state:
        st.session_state.compliance_scores = {}
        
    # Sidebar navigation
    with st.sidebar:
        st.title("AI Compliance Tool")
        st.button("Home", on_click=lambda: set_page("home"))
        st.button("Model Configuration", on_click=lambda: set_page("model_config"))
        
        if st.session_state.model_configured:
            st.button("Test Configuration", on_click=lambda: set_page("test_config"))
            st.button("Run Tests", on_click=lambda: set_page("run_tests"))
            st.button("Results Dashboard", on_click=lambda: set_page("results"))
            
        st.divider()
        
        if st.session_state.model_configured:
            st.success("Model Configured")
            st.info(f"Type: {st.session_state.model_type}")
            
        st.markdown("### About")
        st.info("This tool helps ensure your AI models comply with regulatory and ethical standards.")
    
    # Main content based on current page
    if st.session_state.page == "home":
        render_home_page()
    elif st.session_state.page == "model_config":
        render_model_config_page()
    elif st.session_state.page == "test_config":
        render_test_config_page()
    elif st.session_state.page == "run_tests":
        render_run_tests_page()
    elif st.session_state.page == "results":
        render_results_page()

if __name__ == "__main__":
    main()
