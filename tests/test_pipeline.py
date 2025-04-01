#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Unit tests for the Retail GenAI Accelerator inference pipeline.
"""

import os
import sys
import unittest
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from unittest.mock import patch, MagicMock

# Add parent directory to path for importing project modules
sys.path.append(str(Path(__file__).parent.parent))

# Import the pipeline class - handle import error for CI environment
try:
    from src.inference.pipeline import RetailGenAIPipeline
    PIPELINE_AVAILABLE = True
except ImportError:
    PIPELINE_AVAILABLE = False


@unittest.skipIf(not PIPELINE_AVAILABLE, "Pipeline module not available")
class TestInferencePipeline(unittest.TestCase):
    """Test cases for the inference pipeline."""

    def setUp(self):
        """Set up the test environment."""
        # Create mock components
        self.mock_vision_model = MagicMock()
        self.mock_vision_model.return_value = torch.ones((1, 2048))
        
        self.mock_language_model = MagicMock()
        self.mock_language_model.return_value = MagicMock()
        self.mock_language_model.return_value.last_hidden_state = torch.ones((1, 1, 384))
        
        self.mock_tokenizer = MagicMock()
        self.mock_tokenizer.return_value = {
            "input_ids": torch.zeros((1, 128), dtype=torch.long),
            "attention_mask": torch.ones((1, 128))
        }
        
        self.mock_classifier = MagicMock()
        self.mock_classifier.return_value = torch.tensor([[0.1, 0.2, 0.7, 0.05, 0.05]])
        
        self.transforms = MagicMock()
        self.transforms.return_value = torch.ones((3, 224, 224))
        
        self.category_mapping = {
            0: "Electronics",
            1: "Clothing",
            2: "Groceries",
            3: "Home",
            4: "Beauty"
        }
        
        # Create pipeline
        self.pipeline = RetailGenAIPipeline(
            vision_model=self.mock_vision_model,
            language_model=self.mock_language_model,
            tokenizer=self.mock_tokenizer,
            classifier=self.mock_classifier,
            transforms=self.transforms,
            category_mapping=self.category_mapping,
            device="cpu"
        )
        
        # Create a dummy image
        self.dummy_image = Image.new("RGB", (100, 100), color="red")

    def test_pipeline_initialization(self):
        """Test if the pipeline can be initialized correctly."""
        self.assertIsInstance(self.pipeline, RetailGenAIPipeline)
        self.assertEqual(self.pipeline.device, "cpu")
        self.assertEqual(self.pipeline.category_mapping, self.category_mapping)

    @patch('src.inference.pipeline.torch.no_grad')
    def test_predict(self, mock_no_grad):
        """Test the prediction functionality."""
        # Configure mocks
        mock_no_grad.return_value.__enter__ = MagicMock()
        mock_no_grad.return_value.__exit__ = MagicMock()
        
        # Call predict
        result = self.pipeline.predict(self.dummy_image, "Test product")
        
        # Check result structure
        self.assertIn("predicted_category", result)
        self.assertIn("confidence", result)
        self.assertIn("all_probabilities", result)
        self.assertIn("inference_time", result)
        
        # Check values based on our mock setup
        self.assertEqual(result["predicted_category"], "Groceries")  # Index 2 has highest value
        self.assertAlmostEqual(result["confidence"], 0.7)
        self.assertEqual(len(result["all_probabilities"]), 5)

    def test_process_shelf_image(self):
        """Test the shelf image processing."""
        result = self.pipeline.process_shelf_image(self.dummy_image)
        
        # Check result structure
        self.assertIn("num_products", result)
        self.assertIn("detections", result)
        self.assertIn("shelf_image_size", result)
        
        # Each detection should have these keys
        if result["detections"]:
            detection = result["detections"][0]
            self.assertIn("box", detection)
            self.assertIn("category", detection)
            self.assertIn("confidence", detection)
            self.assertIn("product_id", detection)

    def test_answer_product_question(self):
        """Test the question answering functionality."""
        with patch.object(self.pipeline, 'predict') as mock_predict:
            # Configure the mock
            mock_predict.return_value = {
                "predicted_category": "Electronics",
                "confidence": 0.9,
                "all_probabilities": {"Electronics": 0.9, "Clothing": 0.05, "Groceries": 0.02, "Home": 0.02, "Beauty": 0.01}
            }
            
            # Call the function
            result = self.pipeline.answer_product_question(self.dummy_image, "What is the price?")
            
            # Check result structure
            self.assertIn("question", result)
            self.assertIn("answer", result)
            self.assertIn("predicted_category", result)
            self.assertIn("confidence", result)
            
            # Verify the answer contains price information for Electronics
            self.assertIn("price", result["answer"].lower())
            self.assertIn("electronics", result["answer"].lower())


if __name__ == "__main__":
    unittest.main()
