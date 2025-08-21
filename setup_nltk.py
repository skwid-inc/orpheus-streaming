#!/usr/bin/env python3
"""
Setup script to download required NLTK data for the TTS server.
Run this after installing dependencies but before starting the server.
"""

import nltk
import sys

def download_nltk_data():
    """Download required NLTK data packages."""
    required_packages = [
        'averaged_perceptron_tagger_eng',
        'punkt',  # Often needed for tokenization
    ]
    
    print("Downloading required NLTK data packages...")
    
    for package in required_packages:
        try:
            nltk.data.find(f'tokenizers/{package}' if package == 'punkt' else f'taggers/{package}')
            print(f"✓ {package} already downloaded")
        except LookupError:
            print(f"↓ Downloading {package}...")
            nltk.download(package)
            print(f"✓ {package} downloaded successfully")
    
    print("\nAll NLTK data packages are ready!")

if __name__ == "__main__":
    try:
        download_nltk_data()
    except Exception as e:
        print(f"Error downloading NLTK data: {e}", file=sys.stderr)
        sys.exit(1)
