#!/usr/bin/env bash
# Install Tesseract OCR system dependency on Render
apt-get update -qq && apt-get install -y -qq tesseract-ocr tesseract-ocr-eng

# Install Python dependencies
pip install -r requirements.txt
