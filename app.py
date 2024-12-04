import streamlit as st
from matplotlib import pyplot as plt
from matplotlib import font_manager as fm
from ultralytics import YOLO
from PIL import Image
import numpy as np

# Set up the Streamlit app
st.title("Tulu Script Object Detection")

# Load the Tulu font
tulu_font_path = r"C:\tulu_script_crop_image\Baravu 2.otf"  # Replace with the correct path to your .otf file
tulu_font = fm.FontProperties(fname=tulu_font_path)

# Load the trained YOLO model
model_path = r"C:\Users\swath\OneDrive\Desktop\tulu\models\best (3).pt"  # Replace with your trained model path
model = YOLO(model_path)

# File uploader for the image
uploaded_file = st.file_uploader("Upload an image for prediction", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Convert the uploaded file to a format usable by PIL
    image = Image.open(uploaded_file)
    image = image.convert("RGB")  # Ensure it's in RGB format

    # Predict on the uploaded image
    results = model.predict(source=np.array(image), device="cpu", save=False)
    result = results[0]

    # Plot the image with annotations
    fig, ax = plt.subplots(figsize=(10, 10))

    # Display the base image
    ax.imshow(image)

    if result.boxes:  # Check if any detections exist
        for box in result.boxes:
            # Bounding box coordinates
            x1, y1, x2, y2 = box.xyxy[0]
            width, height = x2 - x1, y2 - y1

            # Predicted class and confidence
            cls = int(box.cls[0])
            conf = box.conf[0]
            tulu_label = str(cls)  # Convert the predicted value to string

            # Draw the bounding box
            rect = plt.Rectangle((x1, y1), width, height, linewidth=2, edgecolor="red", facecolor="none")
            ax.add_patch(rect)

            # Add the label in Tulu font above the bounding box
            ax.text(
                x1, y1 - 5,  # Position above the top-left corner of the bounding box
                f"{tulu_label}",
                fontproperties=tulu_font,
                fontsize=20,
                color="blue",
                bbox=dict(facecolor="yellow", alpha=0.5, edgecolor="none")  # Background for readability
            )
    else:
        ax.text(
            0.5, 0.5,
            "No objects detected.",
            fontproperties=tulu_font,
            fontsize=16,
            ha="center",
            va="center",
            transform=ax.transAxes
        )

    # Turn off axes
    ax.axis("off")

    # Display the annotated image
    st.pyplot(fig)
else:
    st.info("Please upload an image to start the prediction.")
