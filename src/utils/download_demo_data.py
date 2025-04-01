#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Download and prepare demo data for the NVIDIA GenAI Retail Accelerator project.
"""

import os
import sys
import argparse
import requests
import zipfile
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm


def download_file(url, destination):
    """
    Download a file from a URL to a destination path with progress bar.
    
    Args:
        url (str): URL to download
        destination (str or Path): Destination file path
    
    Returns:
        Path: Path to the downloaded file
    """
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    file_size = int(response.headers.get('content-length', 0))
    block_size = 1024
    progress_bar = tqdm(total=file_size, unit='iB', unit_scale=True, desc=f"Downloading {destination.name}")
    
    with open(destination, 'wb') as f:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            f.write(data)
    
    progress_bar.close()
    return destination


def extract_zip(zip_path, extract_to):
    """
    Extract a zip file to a directory.
    
    Args:
        zip_path (str or Path): Path to the zip file
        extract_to (str or Path): Directory to extract to
    """
    extract_to = Path(extract_to)
    extract_to.mkdir(parents=True, exist_ok=True)
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        total_files = len(zip_ref.infolist())
        with tqdm(total=total_files, desc=f"Extracting {zip_path.name}") as pbar:
            for file in zip_ref.infolist():
                zip_ref.extract(file, extract_to)
                pbar.update(1)


def generate_sample_data(output_dir, num_products=100, num_sales=1000):
    """
    Generate sample retail data if no real data is available.
    
    Args:
        output_dir (str or Path): Directory to save the generated data
        num_products (int): Number of products to generate
        num_sales (int): Number of sales transactions to generate
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate product catalog
    product_data = {
        "product_id": list(range(1, num_products+1)),
        "name": [f"Product {i}" for i in range(1, num_products+1)],
        "category": np.random.choice([
            "Electronics", "Clothing", "Groceries", "Home", "Beauty", 
            "Toys", "Sports", "Automotive", "Health", "Pet Supplies"
        ], num_products),
        "price": np.random.uniform(5, 500, num_products).round(2),
        "description": [
            f"This is a detailed description for product {i}. "
            f"It has unique features and benefits for customers." 
            for i in range(1, num_products+1)
        ],
        "in_stock": np.random.choice([True, False], num_products, p=[0.8, 0.2]),
        "stock_quantity": np.random.randint(0, 100, num_products)
    }
    
    # Create DataFrame and save to CSV
    product_df = pd.DataFrame(product_data)
    product_df.to_csv(output_dir / "product_catalog.csv", index=False)
    print(f"Generated product catalog with {num_products} products")
    
    # Generate sales data
    customer_ids = list(range(1, 101))
    dates = pd.date_range(start='2024-01-01', end='2024-03-31')
    
    sales_data = []
    for _ in range(num_sales):
        customer_id = np.random.choice(customer_ids)
        date = np.random.choice(dates)
        product_id = np.random.choice(product_data["product_id"])
        price = product_df.loc[product_df["product_id"] == product_id, "price"].iloc[0]
        quantity = np.random.randint(1, 5)
        
        sales_data.append({
            "sale_id": _ + 1,
            "customer_id": customer_id,
            "product_id": product_id,
            "date": date.strftime("%Y-%m-%d"),
            "quantity": quantity,
            "unit_price": price,
            "total_price": (price * quantity).round(2)
        })
    
    # Create DataFrame and save to CSV
    sales_df = pd.DataFrame(sales_data)
    sales_df.to_csv(output_dir / "sales_transactions.csv", index=False)
    print(f"Generated sales data with {num_sales} transactions")
    
    # Generate store layout data
    store_sections = ["Entrance", "Checkout", "Electronics", "Clothing", "Groceries", 
                     "Home", "Beauty", "Toys", "Sports", "Automotive", "Health", "Pet Supplies"]
    
    layout_data = []
    x, y = 0, 0
    for section in store_sections:
        width = np.random.randint(3, 8)
        height = np.random.randint(3, 8)
        layout_data.append({
            "section_name": section,
            "x_position": x,
            "y_position": y,
            "width": width,
            "height": height,
            "adjacent_sections": []
        })
        x += width + 1
        if x > 20:
            x = 0
            y += height + 1
    
    # Add adjacency information
    for i, section in enumerate(layout_data):
        adjacent = []
        for j, other in enumerate(layout_data):
            if i == j:
                continue
            
            # Check if sections are adjacent
            x1, y1 = section["x_position"], section["y_position"]
            x2, y2 = other["x_position"], other["y_position"]
            w1, h1 = section["width"], section["height"]
            w2, h2 = other["width"], other["height"]
            
            # Simple adjacency check
            if ((abs(x1 + w1 - x2) <= 1 and (y1 <= y2 + h2 and y2 <= y1 + h1)) or
                (abs(y1 + h1 - y2) <= 1 and (x1 <= x2 + w2 and x2 <= x1 + w1))):
                adjacent.append(other["section_name"])
        
        section["adjacent_sections"] = adjacent
    
    layout_df = pd.DataFrame(layout_data)
    layout_df.to_csv(output_dir / "store_layout.csv", index=False)
    print(f"Generated store layout data with {len(store_sections)} sections")
    
    return product_df, sales_df, layout_df


def main():
    """Main function to download and prepare demo data."""
    parser = argparse.ArgumentParser(description="Download demo data for the NVIDIA GenAI Retail Accelerator")
    parser.add_argument("--data-dir", type=str, default="../examples/product_data",
                        help="Directory to save the data (default: ../examples/product_data)")
    parser.add_argument("--generate-only", action="store_true",
                        help="Only generate sample data without attempting to download real data")
    parser.add_argument("--products", type=int, default=100,
                        help="Number of products to generate (default: 100)")
    parser.add_argument("--sales", type=int, default=1000,
                        help="Number of sales transactions to generate (default: 1000)")
    
    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    
    print("Preparing demo data for NVIDIA GenAI Retail Accelerator...")
    
    if not args.generate_only:
        print("Note: This script would normally download real retail datasets.")
        print("For this demo, we'll generate synthetic data instead.")
    
    # Generate sample data
    generate_sample_data(data_dir, args.products, args.sales)
    
    print("\nDemo data preparation complete!")
    print(f"Data saved to: {data_dir.absolute()}")
    print("\nYou can now use this data in the notebooks.")


if __name__ == "__main__":
    main()
