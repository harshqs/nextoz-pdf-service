#!/usr/bin/env bash
apt-get update -qq && apt-get install -y -qq tesseract-ocr tesseract-ocr-eng
pip install -r requirements.txt
