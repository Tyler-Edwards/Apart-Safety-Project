"""
Model initializer for testing purposes.
This module provides functions to initialize pretrained models for each model type.
"""

import streamlit as st
from transformers import (
    AutoModelForSequenceClassification, 
    AutoTokenizer,
    AutoModelForImageClassification,
    AutoImageProcessor,
    AutoModelForAudioClassification,
    AutoFeatureExtractor,
    AutoModelForCausalLM,
    pipeline
)

# Map of model types to pretrained models
MODEL_MAP = {
    # NLP Models
    "Text Classification": {
        "model_id": "distilbert-base-uncased-finetuned-sst-2-english",
        "type": "text-classification"
    },
    "Token Classification": {
        "model_id": "dbmdz/bert-large-cased-finetuned-conll03-english",
        "type": "token-classification"
    },
    "Question Answering": {
        "model_id": "distilbert-base-cased-distilled-squad",
        "type": "question-answering"
    },
    "Text Generation": {
        "model_id": "gpt2",
        "type": "text-generation"
    },
    # Text Generation Models
    "Chat Completion": {
        "model_id": "gpt2",
        "type": "text-generation"
    },
    "Instruction Following": {
        "model_id": "EleutherAI/gpt-neo-1.3B",
        "type": "text-generation"
    },
    "Creative Writing": {
        "model_id": "gpt2-medium",
        "type": "text-generation"
    },
    "Conversational AI": {
        "model_id": "facebook/blenderbot-400M-distill",
        "type": "text-generation"
    },
    "Content Generation": {
        "model_id": "gpt2-large",
        "type": "text-generation"
    },
    
    # Vision Models
    "Image Classification": {
        "model_id": "google/vit-base-patch16-224",
        "type": "image-classification"
    },
    "Object Detection": {
        "model_id": "facebook/detr-resnet-50",
        "type": "object-detection"
    },
    
    # Audio Models
    "Audio Classification": {
        "model_id": "superb/hubert-base-superb-ks",
        "type": "audio-classification"
    },
    
    # Multimodal Models
    "Visual Question Answering": {
        "model_id": "dandelin/vilt-b32-finetuned-vqa",
        "type": "vqa"
    },
    
    # Default/Fallback model
    "default": {
        "model_id": "distilbert-base-uncased",
        "type": "text-classification"
    }
}

def get_model_info(model_type):
    """
    Get model information for the specified model type.
    
    Args:
        model_type (str): The type of model to initialize
        
    Returns:
        dict: Dictionary containing model_id and model_type
    """
    return MODEL_MAP.get(model_type, MODEL_MAP["default"])

@st.cache_resource
def initialize_model(model_type):
    """
    Initialize a pretrained model for the specified model type.
    This function is cached to avoid reloading models unnecessarily.
    
    Args:
        model_type (str): The type of model to initialize
        
    Returns:
        tuple: (model, tokenizer/processor, pipeline)
    """
    model_info = get_model_info(model_type)
    model_id = model_info["model_id"]
    pipeline_type = model_info["type"]
    
    try:
        st.write(f"Initializing {model_type} model: {model_id}")
        
        # Special handling for text generation models
        if pipeline_type == "text-generation":
            from transformers import AutoModelForCausalLM
            
            # Set max_length for tokenizer
            max_length = 512
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForCausalLM.from_pretrained(model_id)
            
            # Set generation parameters
            gen_kwargs = {
                "max_length": max_length,
                "do_sample": True,
                "temperature": 0.7,
                "top_p": 0.9,
                "pad_token_id": tokenizer.eos_token_id
            }
            
            # Create generation pipeline
            text_gen_pipeline = pipeline(
                pipeline_type, 
                model=model, 
                tokenizer=tokenizer,
                **gen_kwargs
            )
            
            return model, tokenizer, text_gen_pipeline
            
        # For other text models
        elif pipeline_type in ["text-classification", "token-classification", 
                            "question-answering", "summarization"]:
            # Set max_length for tokenizer
            max_length = 512
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForSequenceClassification.from_pretrained(model_id)
            
            # Create pipeline with explicit max_length
            nlp_pipeline = pipeline(
                pipeline_type, 
                model=model, 
                tokenizer=tokenizer,
                max_length=max_length,
                truncation=True
            )
            return model, tokenizer, nlp_pipeline
            
        # For image models
        elif pipeline_type in ["image-classification", "object-detection"]:
            processor = AutoImageProcessor.from_pretrained(model_id)
            model = AutoModelForImageClassification.from_pretrained(model_id)
            image_pipeline = pipeline(pipeline_type, model=model, image_processor=processor)
            return model, processor, image_pipeline
            
        # For audio models
        elif pipeline_type in ["audio-classification"]:
            feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)
            model = AutoModelForAudioClassification.from_pretrained(model_id)
            audio_pipeline = pipeline(pipeline_type, model=model, feature_extractor=feature_extractor)
            return model, feature_extractor, audio_pipeline
            
        # For VQA or other multimodal tasks
        elif pipeline_type in ["vqa"]:
            # VQA typically needs special handling, use default pipeline
            multimodal_pipeline = pipeline(pipeline_type, model=model_id)
            return None, None, multimodal_pipeline
            
        # Default fallback
        else:
            tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
            model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased")
            default_pipeline = pipeline("text-classification", model=model, tokenizer=tokenizer)
            return model, tokenizer, default_pipeline
            
    except Exception as e:
        st.error(f"Error initializing model: {str(e)}")
        st.warning("Falling back to default model")
        
        # Fallback to a simple model that's very likely to work
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased")
        fallback_pipeline = pipeline("text-classification", model=model, tokenizer=tokenizer)
        return model, tokenizer, fallback_pipeline

def extend_model_adapter(model_adapter, model_type):
    """
    Extends the ModelAdapter with a real pretrained model.
    
    Args:
        model_adapter: The ModelAdapter instance to extend
        model_type: The type of model to initialize
        
    Returns:
        ModelAdapter: The extended ModelAdapter instance
    """
    model, tokenizer_or_processor, nlp_pipeline = initialize_model(model_type)
    
    # Update the ModelAdapter with the real model
    model_adapter.model = model
    model_adapter.tokenizer = tokenizer_or_processor if hasattr(model, "config") else None
    model_adapter.processor = tokenizer_or_processor if not hasattr(model, "config") else None
    model_adapter.pipeline = nlp_pipeline
    
    return model_adapter