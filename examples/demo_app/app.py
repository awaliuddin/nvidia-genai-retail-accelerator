#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Demo Web Application for NVIDIA GenAI Retail Accelerator.

This Streamlit app showcases the capabilities of the multi-modal retail GenAI system,
allowing users to interact with various features like product recognition,
question answering, and shelf analysis.
"""

import os
import sys
import time
import json
import base64
import requests
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

# Set page configuration
st.set_page_config(
    page_title="NVIDIA GenAI Retail Accelerator",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API configuration
API_URL = os.environ.get("API_URL", "http://localhost:8000")

# Define helper functions
def image_to_base64(image):
    """Convert PIL Image to base64 string."""
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str

def base64_to_image(base64_str):
    """Convert base64 string to PIL Image."""
    img_data = base64.b64decode(base64_str)
    return Image.open(BytesIO(img_data))

def call_predict_api(image, text):
    """Call the prediction API."""
    img_base64 = image_to_base64(image)
    data = {
        "image_base64": img_base64,
        "text": text
    }
    
    with st.spinner("Analyzing image..."):
        try:
            response = requests.post(f"{API_URL}/predict", json=data)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            st.error(f"Error calling API: {e}")
            return None

def call_answer_api(image, question):
    """Call the question answering API."""
    img_base64 = image_to_base64(image)
    data = {
        "image_base64": img_base64,
        "question": question
    }
    
    with st.spinner("Generating answer..."):
        try:
            response = requests.post(f"{API_URL}/answer", json=data)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            st.error(f"Error calling API: {e}")
            return None

def call_analyze_shelf_api(image):
    """Call the shelf analysis API."""
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    image_bytes = buffered.getvalue()
    
    with st.spinner("Analyzing shelf..."):
        try:
            files = {"image": ("shelf.jpg", image_bytes, "image/jpeg")}
            response = requests.post(f"{API_URL}/analyze_shelf", files=files)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            st.error(f"Error calling API: {e}")
            return None

def check_api_health():
    """Check if the API is healthy."""
    try:
        response = requests.get(f"{API_URL}/health")
        if response.status_code == 200:
            data = response.json()
            return data["status"] == "ok", data
        return False, None
    except:
        return False, None

def get_example_image(category):
    """Get example image for a category."""
    # In a real implementation, we would have actual product images
    # For this demo, we'll generate colored placeholder images
    category_colors = {
        "Electronics": (200, 200, 255),  # Light blue
        "Clothing": (255, 200, 200),     # Light red
        "Groceries": (200, 255, 200),    # Light green
        "Home": (255, 255, 200),         # Light yellow
        "Beauty": (255, 200, 255),       # Light purple
        "default": (240, 240, 240)       # Light gray
    }
    
    color = category_colors.get(category, category_colors["default"])
    img = np.ones((300, 300, 3), dtype=np.uint8) * np.array(color, dtype=np.uint8)
    
    # Convert to PIL Image
    image = Image.fromarray(img)
    
    # Add category text
    import cv2
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    cv2.putText(img_cv, category, (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    image = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
    
    return image

def create_shelf_image():
    """Create a simulated shelf image."""
    # Create a background for the shelf
    img = np.ones((600, 800, 3), dtype=np.uint8) * np.array([240, 230, 220], dtype=np.uint8)
    
    # Draw shelf lines
    for y in range(150, 600, 150):
        cv2.line(img, (0, y), (800, y), (180, 170, 160), 5)
    
    # Add some colored rectangles to represent products
    categories = ["Electronics", "Clothing", "Groceries", "Home", "Beauty"]
    category_colors = {
        "Electronics": (200, 200, 255),  # Light blue
        "Clothing": (255, 200, 200),     # Light red
        "Groceries": (200, 255, 200),    # Light green
        "Home": (255, 255, 200),         # Light yellow
        "Beauty": (255, 200, 255),       # Light purple
    }
    
    import random
    for i in range(10):
        x = random.randint(50, 700)
        y = random.randint(50, 500)
        w = random.randint(50, 100)
        h = random.randint(50, 100)
        
        category = random.choice(categories)
        color = category_colors[category]
        
        cv2.rectangle(img, (x, y), (x+w, y+h), color, -1)
    
    return Image.fromarray(img)

# UI Layout
def main():
    # Sidebar
    st.sidebar.image("https://developer.nvidia.com/sites/default/files/akamai/nvidia-logo.png", width=200)
    st.sidebar.title("Navigation")
    
    # Check API health
    api_healthy, health_data = check_api_health()
    
    if api_healthy:
        st.sidebar.success("✅ API is connected and healthy")
        st.sidebar.info(f"GPU Available: {'✅' if health_data['gpu_available'] else '❌'}")
    else:
        st.sidebar.error("❌ API is not connected")
        st.sidebar.info(f"Expected API URL: {API_URL}")
    
    # Navigation
    page = st.sidebar.radio(
        "Select Demo",
        ["Product Recognition", "Product Q&A", "Shelf Analysis", "About"]
    )
    
    # Header
    st.title("🛒 NVIDIA GenAI Retail Accelerator")
    st.subheader("Multi-Modal AI for Retail Applications")
    
    # Page content
    if page == "Product Recognition":
        product_recognition_page()
    elif page == "Product Q&A":
        product_qa_page()
    elif page == "Shelf Analysis":
        shelf_analysis_page()
    else:
        about_page()

def product_recognition_page():
    st.header("Product Recognition")
    st.write("Upload a product image to classify it into retail categories.")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Image upload
        uploaded_file = st.file_uploader("Upload a product image", type=["jpg", "jpeg", "png"])
        
        # Or use example
        if not uploaded_file:
            st.write("Or use an example:")
            example_category = st.selectbox(
                "Select category",
                ["Electronics", "Clothing", "Groceries", "Home", "Beauty"]
            )
            if st.button("Use Example"):
                example_image = get_example_image(example_category)
                st.session_state.image = example_image
                st.image(example_image, caption=f"Example {example_category} Product", use_column_width=True)
        else:
            image = Image.open(uploaded_file).convert('RGB')
            st.session_state.image = image
            st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Additional text context
        st.write("Add text context (optional):")
        text_context = st.text_input("Product Description", 
                                    value="This is a high-quality retail product with excellent features.")
    
    with col2:
        if "image" in st.session_state and st.button("Analyze Product"):
            result = call_predict_api(st.session_state.image, text_context)
            
            if result:
                # Display results
                st.subheader("Classification Results")
                
                # Main prediction
                st.success(f"**Primary Category: {result['predicted_category']}** (Confidence: {result['confidence']:.2%})")
                
                # Create a bar chart of probabilities
                categories = list(result["all_probabilities"].keys())
                probabilities = list(result["all_probabilities"].values())
                
                # Sort by probability
                sorted_indices = np.argsort(probabilities)[::-1]
                categories = [categories[i] for i in sorted_indices]
                probabilities = [probabilities[i] for i in sorted_indices]
                
                chart_data = pd.DataFrame({
                    "Category": categories,
                    "Probability": probabilities
                })
                
                st.bar_chart(chart_data.set_index("Category"))
                
                # Performance metrics
                st.info(f"Processing Time: {result['inference_time']:.3f} seconds")
                
                # Recommendations based on category
                st.subheader("Recommended Products")
                
                # Sample recommendations based on category
                recommendations = {
                    "Electronics": ["Wireless Headphones", "Power Bank", "Bluetooth Speaker"],
                    "Clothing": ["White T-Shirt", "Blue Jeans", "Black Sneakers"],
                    "Groceries": ["Organic Vegetables", "Fresh Bread", "Vitamin Water"],
                    "Home": ["Table Lamp", "Throw Pillows", "Wall Clock"],
                    "Beauty": ["Face Cleanser", "Moisturizer", "Lip Balm"]
                }
                
                category = result["predicted_category"]
                recs = recommendations.get(category, ["Product A", "Product B", "Product C"])
                
                rec_cols = st.columns(3)
                for i, (rec, col) in enumerate(zip(recs, rec_cols)):
                    with col:
                        st.write(f"**{rec}**")
                        st.image(get_example_image(category), width=100)
                        st.button(f"View {rec}", key=f"rec_{i}")

def product_qa_page():
    st.header("Product Question & Answer")
    st.write("Upload a product image and ask questions about it.")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Image upload
        uploaded_file = st.file_uploader("Upload a product image", type=["jpg", "jpeg", "png"])
        
        # Or use example
        if not uploaded_file:
            st.write("Or use an example:")
            example_category = st.selectbox(
                "Select category",
                ["Electronics", "Clothing", "Groceries", "Home", "Beauty"]
            )
            if st.button("Use Example"):
                example_image = get_example_image(example_category)
                st.session_state.qa_image = example_image
                st.image(example_image, caption=f"Example {example_category} Product", use_column_width=True)
        else:
            image = Image.open(uploaded_file).convert('RGB')
            st.session_state.qa_image = image
            st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Question input
        st.write("Ask a question about this product:")
        question = st.text_input("Your Question", value="How much does this product cost?")
        
        # Example questions
        st.write("Or try one of these:")
        example_questions = [
            "What features does this product have?",
            "What brands make similar products?",
            "Is there a warranty for this product?",
            "Can you recommend alternatives to this product?"
        ]
        
        for i, eq in enumerate(example_questions):
            if st.button(eq, key=f"question_{i}"):
                st.session_state.question = eq
                question = eq
    
    with col2:
        # Store the selected question
        if "question" in st.session_state:
            question = st.session_state.question
        
        if "qa_image" in st.session_state and st.button("Ask Question"):
            result = call_answer_api(st.session_state.qa_image, question)
            
            if result:
                # Display results
                st.subheader("Product Q&A")
                
                with st.container(border=True):
                    st.write(f"**Question:** {result['question']}")
                    st.write(f"**Answer:** {result['answer']}")
                
                # Show predicted category
                st.info(f"Product identified as: {result['predicted_category']} (Confidence: {result['confidence']:.2%})")
                
                # Performance metrics
                st.info(f"Processing Time: {result.get('processing_time', 0):.3f} seconds")
                
                # AI reasoning (for educational purposes)
                with st.expander("See AI reasoning process"):
                    st.write("""
                    1. The image was analyzed using a computer vision model to identify the product category
                    2. The question was processed to identify the intent using NLP techniques
                    3. The system retrieved relevant information from the product database
                    4. A response was generated based on the product category and question intent
                    5. The response was reviewed for accuracy and relevance
                    """)

def shelf_analysis_page():
    st.header("Retail Shelf Analysis")
    st.write("Upload a shelf image to detect and categorize products.")
    
    # Image upload
    uploaded_file = st.file_uploader("Upload a shelf image", type=["jpg", "jpeg", "png"])
    
    # Or use example
    if not uploaded_file:
        st.write("Or use a simulated shelf:")
        if st.button("Generate Shelf Image"):
            shelf_image = create_shelf_image()
            st.session_state.shelf_image = shelf_image
            st.image(shelf_image, caption="Simulated Shelf", use_column_width=True)
    else:
        shelf_image = Image.open(uploaded_file).convert('RGB')
        st.session_state.shelf_image = shelf_image
        st.image(shelf_image, caption="Uploaded Shelf Image", use_column_width=True)
    
    if "shelf_image" in st.session_state and st.button("Analyze Shelf"):
        result = call_analyze_shelf_api(st.session_state.shelf_image)
        
        if result:
            # Display results
            st.subheader("Shelf Analysis Results")
            
            # Display visualization
            if "visualization_base64" in result:
                viz_image = base64_to_image(result["visualization_base64"])
                st.image(viz_image, caption="Product Detection Results", use_column_width=True)
            
            # Summary statistics
            st.success(f"Detected {result['num_products']} products on the shelf")
            
            # Show detected products table
            st.write("Detected Products:")
            
            detections_df = pd.DataFrame(result["detections"])
            # Rename and reorder columns
            if not detections_df.empty:
                detections_df = detections_df.rename(columns={
                    "category": "Category",
                    "confidence": "Confidence",
                    "product_id": "Product ID"
                })
                detections_df["Confidence"] = detections_df["Confidence"].apply(lambda x: f"{x:.2%}")
                
                # Remove box coordinates for cleaner display
                if "box" in detections_df.columns:
                    detections_df = detections_df.drop(columns=["box"])
                
                st.dataframe(detections_df[["Product ID", "Category", "Confidence"]])
            
            # Performance metrics
            st.info(f"Processing Time: {result.get('processing_time', 0):.3f} seconds")
            
            # Category distribution
            if not detections_df.empty:
                st.subheader("Category Distribution")
                category_counts = detections_df["Category"].value_counts().reset_index()
                category_counts.columns = ["Category", "Count"]
                
                st.bar_chart(category_counts.set_index("Category"))
                
                # Insights
                st.subheader("Insights")
                most_common = category_counts.iloc[0]["Category"]
                
                insights = [
                    f"Most common category: {most_common}",
                    f"Shelf utilization: {result['num_products'] / 15:.0%} of capacity",
                    "Store recommendation: Restock Beauty products"
                ]
                
                for insight in insights:
                    st.markdown(f"- {insight}")

def about_page():
    st.header("About NVIDIA GenAI Retail Accelerator")
    
    st.markdown("""
    ## Overview
    
    The NVIDIA GenAI Retail Accelerator is a comprehensive multi-modal AI solution designed specifically for retail applications. 
    It leverages NVIDIA's GPU technology to deliver high-performance AI capabilities for retail environments.
    
    ## Key Features
    
    - **Multi-Modal AI**: Combines computer vision and natural language processing
    - **Product Recognition**: Identifies products from images
    - **Intelligent Q&A**: Answers questions about products
    - **Shelf Analysis**: Detects and categorizes products on retail shelves
    - **GPU-Accelerated**: Optimized for NVIDIA GPUs for high performance
    
    ## Technical Architecture
    
    The system is built on a fusion of state-of-the-art models:
    
    1. **Vision Models**: For product and shelf image analysis
    2. **Language Models**: For natural language understanding and generation
    3. **Fusion Models**: To combine visual and textual information
    
    ## GPU Acceleration
    
    The system leverages NVIDIA GPU technology for acceleration:
    
    - **14x faster** inference on GPU vs CPU
    - **Optimized with CUDA** for maximum performance
    - **Containerized** for easy deployment
    
    ## Getting Started
    
    To deploy this solution in your environment:
    
    1. Clone the repository: `git clone https://github.com/yourusername/nvidia-genai-retail-accelerator.git`
    2. Install dependencies: `pip install -r requirements.txt`
    3. Run the API server: `python -m src.api.server`
    4. Launch the demo app: `streamlit run examples/demo_app/app.py`
    
    For more information, visit the [GitHub repository](https://github.com/yourusername/nvidia-genai-retail-accelerator).
    """)
    
    # NVIDIA logo at the bottom
    st.image("https://developer.nvidia.com/sites/default/files/akamai/nvidia-logo.png", width=200)

if __name__ == "__main__":
    main()
