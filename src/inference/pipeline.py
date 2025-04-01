#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Inference pipeline for the Retail GenAI system.

This module implements an end-to-end inference pipeline that combines
vision and language models for retail applications.
"""

import os
import json
import time
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
import torchvision.transforms as transforms
from transformers import AutoTokenizer, AutoModel
import torchvision.models as vision_models

# Import project modules
from src.models.multimodal_fusion import RetailProductFusionModel

class MultiModalClassifier(nn.Module):
    """Multi-modal classifier combining vision and language features."""
    
    def __init__(self, fusion_model, num_classes):
        super(MultiModalClassifier, self).__init__()
        self.fusion_model = fusion_model
        self.classifier = nn.Linear(fusion_model.output_dim, num_classes)
    
    def forward(self, img_features, text_features):
        outputs = self.fusion_model(img_features=img_features, text_features=text_features)
        embeddings = outputs["embeddings"]
        logits = self.classifier(embeddings)
        return logits

class RetailGenAIPipeline:
    """End-to-end inference pipeline for retail GenAI system."""
    
    def __init__(self, vision_model, language_model, tokenizer, classifier, 
                 transforms, category_mapping, device="cuda"):
        self.vision_model = vision_model
        self.language_model = language_model
        self.tokenizer = tokenizer
        self.classifier = classifier
        self.transforms = transforms
        self.category_mapping = category_mapping
        self.device = device
        
        # Set all models to evaluation mode
        self.vision_model.eval()
        self.language_model.eval()
        self.classifier.eval()
    
    @classmethod
    def from_pretrained(cls, model_dir, device="cuda"):
        """Load a pipeline from pretrained models."""
        model_dir = Path(model_dir)
        
        # Load model metadata
        metadata_path = model_dir / "multimodal_model_metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                model_metadata = json.load(f)
        else:
            # Default metadata for demo purposes
            model_metadata = {
                "model_type": "MultiModalClassifier",
                "fusion_type": "attention",
                "img_feature_dim": 2048,  # For ResNet50
                "text_feature_dim": 384,   # For MiniLM-L6
                "hidden_dim": 512,
                "output_dim": 256,
                "num_classes": 5,  # Default to 5 common retail categories
                "category_mapping": {"0": "Electronics", "1": "Clothing", "2": "Groceries", "3": "Home", "4": "Beauty"},
                "test_accuracy": 0.85,
                "trained_on": "demo_dataset",
                "date_trained": "2023-01-01"
            }
        
        # Fix category mapping keys (JSON converts all keys to strings)
        category_mapping = {int(k): v for k, v in model_metadata["category_mapping"].items()}
        
        # Load models
        # 1. Vision model
        vision_model = vision_models.resnet50(pretrained=True)
        # Remove the classification layer
        vision_model = nn.Sequential(*list(vision_model.children())[:-1])
        vision_model = vision_model.to(device)
        vision_model.eval()
        
        # 2. Language model
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        language_model = AutoModel.from_pretrained(model_name)
        language_model = language_model.to(device)
        language_model.eval()
        
        # 3. Fusion model
        fusion_model = RetailProductFusionModel(
            vision_encoder=None,  # We'll use pre-extracted features
            text_encoder=None,    # We'll use pre-extracted features
            fusion_type=model_metadata["fusion_type"],
            img_feature_dim=model_metadata["img_feature_dim"],
            text_feature_dim=model_metadata["text_feature_dim"],
            hidden_dim=model_metadata["hidden_dim"],
            output_dim=model_metadata["output_dim"]
        )
        
        # 4. Classifier
        classifier = MultiModalClassifier(fusion_model, model_metadata["num_classes"])
        
        # Load trained model if exists
        model_path = model_dir / "best_multimodal_classifier.pth"
        if model_path.exists():
            classifier.load_state_dict(torch.load(model_path, map_location=device))
        
        # Move model to device and set to evaluation mode
        classifier = classifier.to(device)
        classifier.eval()
        
        # Image transformations
        image_transforms = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        return cls(
            vision_model=vision_model,
            language_model=language_model,
            tokenizer=tokenizer,
            classifier=classifier,
            transforms=image_transforms,
            category_mapping=category_mapping,
            device=device
        )
    
    def preprocess_image(self, image):
        """Preprocess image for model input."""
        if isinstance(image, str):
            # Load from path
            image = Image.open(image).convert('RGB')
        elif isinstance(image, bytes):
            # Load from bytes
            image = Image.open(BytesIO(image)).convert('RGB')
        elif not isinstance(image, Image.Image):
            raise ValueError("Image must be a PIL Image, a path string, or bytes")
            
        # Apply transformations
        return self.transforms(image).unsqueeze(0)  # Add batch dimension
    
    def preprocess_text(self, text, max_length=128):
        """Preprocess text for model input."""
        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        return encoding
    
    def extract_features(self, image, text):
        """Extract features from vision and language models."""
        with torch.no_grad():
            # Process image
            img_tensor = self.preprocess_image(image)
            img_tensor = img_tensor.to(self.device)
            img_features = self.vision_model(img_tensor).squeeze(-1).squeeze(-1)
            
            # Process text
            text_encoding = self.preprocess_text(text)
            input_ids = text_encoding["input_ids"].to(self.device)
            attention_mask = text_encoding["attention_mask"].to(self.device)
            text_outputs = self.language_model(input_ids=input_ids, attention_mask=attention_mask)
            text_features = text_outputs.last_hidden_state[:, 0, :]
        
        return img_features, text_features
    
    def predict(self, image, text):
        """Run end-to-end prediction pipeline."""
        start_time = time.time()
        
        # Extract features
        img_features, text_features = self.extract_features(image, text)
        
        # Run classifier
        with torch.no_grad():
            logits = self.classifier(img_features, text_features)
            probs = torch.nn.functional.softmax(logits, dim=1)[0]
            predicted_idx = torch.argmax(probs).item()
            predicted_category = self.category_mapping[predicted_idx]
            confidence = probs[predicted_idx].item()
        
        # Collect all probabilities
        category_probs = {}
        for idx, prob in enumerate(probs):
            category = self.category_mapping[idx]
            category_probs[category] = prob.item()
        
        inference_time = time.time() - start_time
        
        return {
            "predicted_category": predicted_category,
            "confidence": confidence,
            "all_probabilities": category_probs,
            "inference_time": inference_time
        }
    
    def process_shelf_image(self, image, text=""):
        """Process a shelf image to identify products (using a simple detection model)."""
        # In a production implementation, this would use a proper object detection model
        # For this demo, we'll use a simplified approach
        import random
        
        # If image is a path or bytes, convert to PIL Image
        if isinstance(image, str):
            img = Image.open(image).convert('RGB')
        elif isinstance(image, bytes):
            img = Image.open(BytesIO(image)).convert('RGB')
        elif isinstance(image, Image.Image):
            img = image
        else:
            raise ValueError("Image must be a PIL Image, a path string, or bytes")
        
        width, height = img.size
        
        # Simulated product detection for demo purposes
        # In a real implementation, this would use a detection model like YOLO or Faster R-CNN
        num_products = random.randint(3, 8)
        product_detections = []
        
        for i in range(num_products):
            # Generate random box (ensuring they don't go outside the image)
            box_width = random.randint(width // 6, width // 3)
            box_height = random.randint(height // 6, height // 3)
            x = random.randint(0, width - box_width)
            y = random.randint(0, height - box_height)
            
            # Random category and confidence
            cat_idx = random.randint(0, len(self.category_mapping) - 1)
            category = self.category_mapping[cat_idx]
            confidence = random.uniform(0.7, 0.99)
            
            product_detections.append({
                "box": [x, y, x + box_width, y + box_height],
                "category": category,
                "confidence": confidence,
                "product_id": f"P{random.randint(1000, 9999)}"
            })
        
        return {
            "num_products": len(product_detections),
            "detections": product_detections,
            "shelf_image_size": [width, height]
        }
    
    def visualize_shelf_detection(self, image, results):
        """Visualize product detections on a shelf image."""
        # Convert to PIL Image if needed
        if not isinstance(image, Image.Image):
            if isinstance(image, str):
                image = Image.open(image).convert('RGB')
            elif isinstance(image, bytes):
                image = Image.open(BytesIO(image)).convert('RGB')
            elif isinstance(image, np.ndarray):
                image = Image.fromarray(image)
        
        # Create a copy of the image to draw on
        img_draw = image.copy()
        draw = ImageDraw.Draw(img_draw)
        
        # Try to load a font (use default if not available)
        try:
            font = ImageFont.truetype("arial.ttf", 14)
        except IOError:
            font = ImageFont.load_default()
        
        # Color mapping for categories
        category_colors = {
            "Electronics": "blue",
            "Clothing": "red",
            "Groceries": "green",
            "Home": "orange",
            "Beauty": "purple"
        }
        
        # Draw bounding boxes and labels
        for det in results["detections"]:
            box = det["box"]
            category = det["category"]
            confidence = det["confidence"]
            product_id = det["product_id"]
            
            # Get color for category
            color = category_colors.get(category, "gray")
            
            # Draw rectangle
            draw.rectangle(box, outline=color, width=3)
            
            # Draw label background
            label = f"{category} ({confidence:.1%})"
            label_size = draw.textbbox((0, 0), label, font=font)
            label_width = label_size[2] - label_size[0]
            label_height = label_size[3] - label_size[1]
            label_bg = [box[0], box[1] - label_height - 4, box[0] + label_width + 4, box[1]]
            draw.rectangle(label_bg, fill=color)
            
            # Draw label text
            draw.text((box[0] + 2, box[1] - label_height - 2), label, fill="white", font=font)
        
        return img_draw
    
    def answer_product_question(self, image, question):
        """Answer a natural language question about a product."""
        # First, predict the product category
        prediction = self.predict(image, question)
        category = prediction["predicted_category"]
        confidence = prediction["confidence"]
        
        # Mock product catalog (in real implementation, this would query a database)
        product_details = {
            "Electronics": {
                "price_range": "$50-$1200",
                "top_brands": ["TechCorp", "Electra", "DigiLife"],
                "features": ["wireless connectivity", "long battery life", "high resolution display"],
                "warranty": "1-3 years"
            },
            "Clothing": {
                "price_range": "$15-$250",
                "top_brands": ["StyleX", "UrbanFit", "ClassicWear"],
                "features": ["sustainable materials", "comfortable fit", "machine washable"],
                "warranty": "30-day returns"
            },
            "Groceries": {
                "price_range": "$2-$35",
                "top_brands": ["FreshFarms", "OrganicLife", "NatureHarvest"],
                "features": ["organic options", "locally sourced", "no preservatives"],
                "warranty": "satisfaction guarantee"
            },
            "Home": {
                "price_range": "$10-$500",
                "top_brands": ["HomeLux", "ComfortLiving", "ModernSpace"],
                "features": ["durable construction", "stylish design", "easy assembly"],
                "warranty": "1-5 years"
            },
            "Beauty": {
                "price_range": "$8-$150",
                "top_brands": ["GlowUp", "NaturalBeauty", "LuxeSkin"],
                "features": ["cruelty-free", "fragrance-free options", "dermatologist tested"],
                "warranty": "30-day returns"
            }
        }
        
        # Simple rule-based QA logic
        response = ""
        
        # Get product info for the predicted category
        if category in product_details:
            info = product_details[category]
            
            # Very basic keyword matching for demo purposes
            question_lower = question.lower()
            
            if "price" in question_lower or "cost" in question_lower or "how much" in question_lower:
                response = f"This {category} product typically costs in the range of {info['price_range']}."
            
            elif "brand" in question_lower or "who makes" in question_lower or "manufacturer" in question_lower:
                top_brands = ", ".join(info["top_brands"])
                response = f"The top brands in this {category} category include {top_brands}."
            
            elif "feature" in question_lower or "specification" in question_lower or "what can" in question_lower:
                features = ", ".join(info["features"])
                response = f"This {category} product typically offers these features: {features}."
            
            elif "warranty" in question_lower or "guarantee" in question_lower or "return" in question_lower:
                response = f"This {category} product typically comes with a {info['warranty']}."
            
            elif "recommend" in question_lower or "alternative" in question_lower or "similar" in question_lower:
                response = f"Based on this {category} product, I would recommend checking out items from {', '.join(info['top_brands'][:2])}."
            
            else:
                # Generic response for other questions
                response = f"This appears to be a {category} product. It typically costs {info['price_range']} and features {', '.join(info['features'][:2])}."
        else:
            response = "I couldn't identify the product category clearly. Could you provide more information?"
        
        return {
            "question": question,
            "answer": response,
            "predicted_category": category,
            "confidence": confidence
        }
