#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Unit tests for the Retail GenAI Accelerator models.
"""

import os
import sys
import unittest
import torch
from pathlib import Path

# Add parent directory to path for importing project modules
sys.path.append(str(Path(__file__).parent.parent))

from src.models.multimodal_fusion import RetailProductFusionModel


class TestMultiModalFusion(unittest.TestCase):
    """Test cases for the multi-modal fusion model."""

    def setUp(self):
        """Set up the test environment."""
        # Use small dimensions for testing
        self.img_feature_dim = 64
        self.text_feature_dim = 32
        self.hidden_dim = 16
        self.output_dim = 8
        self.batch_size = 2
        
        # Initialize model
        self.model = RetailProductFusionModel(
            vision_encoder=None,
            text_encoder=None,
            fusion_type="concat",
            img_feature_dim=self.img_feature_dim,
            text_feature_dim=self.text_feature_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim
        )
        
        # Create dummy inputs
        self.img_features = torch.randn(self.batch_size, self.img_feature_dim)
        self.text_features = torch.randn(self.batch_size, self.text_feature_dim)

    def test_model_initialization(self):
        """Test if the model can be initialized correctly."""
        self.assertIsInstance(self.model, RetailProductFusionModel)
        self.assertEqual(self.model.fusion_type, "concat")
        self.assertEqual(self.model.img_feature_dim, self.img_feature_dim)
        self.assertEqual(self.model.text_feature_dim, self.text_feature_dim)
        self.assertEqual(self.model.hidden_dim, self.hidden_dim)
        self.assertEqual(self.model.output_dim, self.output_dim)

    def test_forward_pass(self):
        """Test the forward pass of the model."""
        outputs = self.model(img_features=self.img_features, text_features=self.text_features)
        
        # Check output keys
        self.assertIn("embeddings", outputs)
        self.assertIn("logits", outputs)
        self.assertIn("img_features", outputs)
        self.assertIn("text_features", outputs)
        
        # Check output shapes
        self.assertEqual(outputs["embeddings"].shape, (self.batch_size, self.output_dim))
        self.assertEqual(outputs["logits"].shape, (self.batch_size, 1))
        self.assertEqual(outputs["img_features"].shape, (self.batch_size, self.hidden_dim))
        self.assertEqual(outputs["text_features"].shape, (self.batch_size, self.hidden_dim))

    def test_different_fusion_types(self):
        """Test different fusion types."""
        for fusion_type in ["concat", "attention", "gated"]:
            model = RetailProductFusionModel(
                vision_encoder=None,
                text_encoder=None,
                fusion_type=fusion_type,
                img_feature_dim=self.img_feature_dim,
                text_feature_dim=self.text_feature_dim,
                hidden_dim=self.hidden_dim,
                output_dim=self.output_dim
            )
            
            outputs = model(img_features=self.img_features, text_features=self.text_features)
            
            # Ensure we get valid outputs for all fusion types
            self.assertIn("embeddings", outputs)
            self.assertEqual(outputs["embeddings"].shape, (self.batch_size, self.output_dim))


if __name__ == "__main__":
    unittest.main()
