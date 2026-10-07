# Web Demo

This directory contains a basic web-based inference demo for the Keyword Spotting model using ONNX Runtime Web.

## How to use
1. Make sure you have exported the model to ONNX format.
2. Copy the exported `.onnx` model to this directory.
3. Update the `index.html` script to load your specific `.onnx` model.
4. Serve the directory locally. For example, using Python:
   ```bash
   python -m http.server 8000
   ```
5. Open `http://localhost:8000` in your web browser.
