# Tulu Script Object Detection

## Overview
This project is a Streamlit-based web application designed to detect objects written in the Tulu script. It uses a pre-trained  model for object detection and renders bounding boxes with labels in the Tulu script font on uploaded images.

## Features
- Upload an image and perform object detection using a  model.
- Display the results with bounding boxes and labels in the Tulu script.
- Easy-to-use interface powered by Streamlit.

## Requirements
To run the project, you will need:
- Python 3.10
- Required Python libraries (listed below)
- A trained model file (`best (3).pt`)
- Tulu script font file (`Baravu 2.otf`)

### Python Libraries
Install the following Python libraries:
- `streamlit`
- `ultralytics`
- `Pillow`
- `matplotlib`
- `numpy`

You can install these libraries using pip:

```bash
pip install streamlit ultralytics Pillow matplotlib numpy
```

## Setup
1. Clone or download the repository.
2. Place your model (`best (3).pt`) in an accessible path.
3. Place your Tulu font file (`Baravu 2.otf`) in the appropriate directory.
4. Update the file paths in the code:
   - Replace the `tulu_font_path` variable with the path to your `.otf` font file.
   - Replace the `model_path` variable with the path to your model file.

## How to Run
1. Open a terminal or command prompt.
2. Navigate to the project directory.
3. Run the Streamlit app:

   ```bash
   streamlit run app.py
   ```

4. The app will open in your default web browser. If it does not, visit `http://localhost:8501` manually.

## Usage
1. Upload an image file in `.jpg`, `.jpeg`, or `.png` format using the file uploader in the web app.
2. Wait for the model to process the image.
3. View the uploaded image with bounding boxes and labels rendered in the Tulu script.

## Project Structure
```plaintext
.
├── app.py                 # Main application file
├── Baravu 2.otf           # Tulu font file
├── models/
│   └── best (3).pt        # Trained model
```

## Notes
- Ensure that the `Baravu 2.otf` font file and model file are placed correctly and paths are updated in the code.
- The application processes images on the CPU for compatibility; adjust if a GPU is available for faster processing.


