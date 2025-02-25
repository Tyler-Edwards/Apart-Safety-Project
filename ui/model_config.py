import streamlit as st
from models.model_adapter import ModelAdapter, ModelCategories
from models.model_initializer import extend_model_adapter

def on_category_change():
    """
    Handles category change event in the model configuration form.
    """
    # Clear the model type when category changes
    if 'model_type_selector' in st.session_state:
        del st.session_state.model_type_selector

    # Add this before the form
    if 'last_category' not in st.session_state:
        st.session_state.last_category = None

def render_model_config_page():
    """
    Renders the model configuration page.
    """
    st.title("Model Configuration")
    
    # Initialize session state variables
    if "last_category" not in st.session_state:
        st.session_state.last_category = "Multimodal"
    if "current_category" not in st.session_state:
        st.session_state.current_category = "Multimodal"
    
    # Define model options
    model_options = ModelCategories.CATEGORIES
    
    # Development mode toggle for loading real models
    with st.sidebar:
        st.divider()
        st.subheader("Development Options")
        use_real_model = st.checkbox(
            "Load real pretrained models", 
            value=False,
            help="When enabled, will load actual pretrained models from Hugging Face for testing"
        )
        st.session_state.use_real_model = use_real_model
        
        if use_real_model:
            st.info("Real model mode enabled. Models will be loaded from Hugging Face when configured.")
            st.warning("This may take some time and requires additional memory.")
        
    # Select model category and type outside the form
    st.subheader("Model Type Selection")
    
    # Category selection outside the form
    category = st.selectbox(
        "Model Category",
        options=list(model_options.keys()),
        key="category_selector"
    )
    
    # Model type selection using the current category
    model_type = st.selectbox(
        "Model Type",
        options=model_options[category],
        key="model_type_selector"
    )
    
    # Store selections in session state for use in the form
    st.session_state.current_category = category
    st.session_state.current_model_type = model_type
    
    # Debug info if needed
    with st.expander("Debug Info", expanded=False):
        st.write(f"Selected Category: {category}")
        st.write(f"Available Model Types: {model_options[category]}")
        st.write(f"Selected Model Type: {model_type}")
    
    # Now start the form for remaining configuration
    with st.form("model_config_form", clear_on_submit=False):
        st.subheader("Basic Information")
        model_name = st.text_input("Model Name", value="My AI Model")

        # Add category-specific configuration options
        if category == "Vision":
            st.subheader("Vision Model Configuration")
            image_size = st.number_input("Input Image Size", value=224)
            color_mode = st.selectbox("Color Mode", ["RGB", "Grayscale"])
            
        elif category == "NLP":
            st.subheader("NLP Model Configuration")
            max_length = st.number_input("Maximum Sequence Length", value=512)
            tokenizer_type = st.selectbox("Tokenizer Type", ["WordPiece", "BPE", "SentencePiece"])
            
        elif category == "Audio":
            st.subheader("Audio Model Configuration")
            sample_rate = st.number_input("Sample Rate (Hz)", value=16000)
            audio_channels = st.selectbox("Audio Channels", ["Mono", "Stereo"])
            
        elif category == "Tabular":
            st.subheader("Tabular Model Configuration")
            input_features = st.number_input("Number of Input Features", value=10)
        
        # Model Access Section
        st.subheader("Model Access")
        access_type = st.radio("Access Method", ["API Endpoint", "Local Model"])
        
        if access_type == "API Endpoint":
            endpoint_url = st.text_input("API Endpoint URL", "https://api.example.com/v1/predict")
            api_key = st.text_input("API Key (if required)", type="password")
        else:
            model_path = st.text_input("Local Model Path", "/path/to/model")
        
        # Use Case & Risk Profile Section
        st.subheader("Use Case & Risk Profile")
        industry = st.selectbox(
            "Industry",
            ["Healthcare", "Finance", "Retail", "Manufacturing", "Education", "Government", "Other"]
        )
        
        risk_level = st.select_slider(
            "Risk Level",
            options=["Low", "Medium", "High", "Critical"],
            value="Medium",
            help="Higher risk models require more rigorous compliance testing"
        )
        
        # Data Sensitivity Section
        data_sensitivity = st.multiselect(
            "Data Sensitivity (select all that apply)",
            ["Personal Identifiable Information (PII)", 
             "Protected Health Information (PHI)",
             "Financial Data",
             "Biometric Data",
             "Location Data",
             "Children's Data",
             "No Sensitive Data"],
            default=["No Sensitive Data"]
        )
        
        # Deployment Context Section
        st.subheader("Deployment Context")
        deployment_env = st.selectbox(
            "Deployment Environment",
            ["Cloud (Public)", "Cloud (Private)", "On-Premises", "Edge/IoT", "Mobile"]
        )
        
        user_access = st.selectbox(
            "User Access Pattern",
            ["Public-Facing", "Authenticated Users Only", "Internal (Employee-Facing)", "Limited Access (Specific Roles)"]
        )

        # Submit button at the end of the form
        submitted = st.form_submit_button("Save Configuration", type="primary")
        
        # Form processing logic - only executes when the form is submitted
        if submitted:
            with st.spinner("Initializing model configuration..."):
                try:
                    # Store category-specific configurations
                    category_config = {}
                    if category == "Vision":
                        category_config = {
                            "image_size": image_size,
                            "color_mode": color_mode
                        }
                    elif category == "NLP":
                        category_config = {
                            "max_length": max_length,
                            "tokenizer_type": tokenizer_type
                        }
                    elif category == "Audio":
                        category_config = {
                            "sample_rate": sample_rate,
                            "audio_channels": audio_channels
                        }
                    elif category == "Tabular":
                        category_config = {
                            "input_features": input_features
                        }
                    
                    # Initialize model adapter based on access type
                    if access_type == "API Endpoint":
                        model_adapter = ModelAdapter(
                            model_type=st.session_state.current_model_type,
                            model_path=endpoint_url,
                            api_key=api_key
                        )
                    else:
                        model_adapter = ModelAdapter(
                            model_type=st.session_state.current_model_type,
                            model_path=model_path
                        )
                    
                    # For development/testing: Initialize real model based on the selected model type
                    use_real_model = st.session_state.get("use_real_model", False)
                    if use_real_model:
                        with st.spinner(f"Loading pretrained model for {st.session_state.current_model_type}..."):
                            try:
                                model_adapter = extend_model_adapter(model_adapter, st.session_state.current_model_type)
                                st.success(f"Loaded pretrained model for {st.session_state.current_model_type}")
                            except Exception as e:
                                st.error(f"Error loading model: {str(e)}")
                                st.warning("Using mock model instead")
                    
                    # Store configurations in session state
                    st.session_state.model_name = model_name
                    st.session_state.model_category = st.session_state.current_category
                    st.session_state.model_type = st.session_state.current_model_type
                    st.session_state.category_config = category_config
                    st.session_state.industry = industry
                    st.session_state.risk_level = risk_level
                    st.session_state.data_sensitivity = data_sensitivity
                    st.session_state.deployment_env = deployment_env
                    st.session_state.user_access = user_access
                    st.session_state.model_adapter = model_adapter
                    
                    # Mark model as configured
                    st.session_state.model_configured = True
                    
                    # Reset any previous test selections
                    st.session_state.selected_tests = []
                    st.session_state.test_results = {}
                    
                    st.success("Model configuration saved successfully!")
                    st.info("Next: Configure compliance tests for your model")
                    
                except Exception as e:
                    st.error(f"Error during model configuration: {str(e)}")
                    st.error("Please check your configuration settings and try again.")

    # Add navigation button outside the form
    if st.session_state.model_configured:
        st.button("Go to Test Configuration", 
                  key="goto_test_config",
                  on_click=lambda: set_page("test_config"))

def set_page(page_name):
    """
    Changes the current page in the session state.
    """
    st.session_state.page = page_name