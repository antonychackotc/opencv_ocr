import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import tempfile
import easyocr

# Preprocessing for better OCR accuracy
def preprocess_plate(plate_img):
    gray = cv2.cvtColor(plate_img, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(resized)
    _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

# Streamlit UI setup
st.set_page_config(page_title="License Plate OCR", layout="centered")
st.title("🚘 YOLOv8 License Plate Detection + OCR")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="📸 Uploaded Image", use_column_width=True)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp:
        image.save(temp.name)
        image_path = temp.name

    # Load YOLOv8 model
    model = YOLO("runs/detect/train/weights/best.pt")
    results = model(image_path)

    for r in results:
        im_array = r.orig_img.copy()
        im_rgb = cv2.cvtColor(im_array, cv2.COLOR_BGR2RGB)

        boxes = r.boxes.xyxy.cpu().numpy().astype(int)
        reader = easyocr.Reader(['en'])

        st.subheader("🔍 Extracted License Plate Texts:")
        all_texts = []

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            pad = 10
            x1 = max(0, x1 - pad)
            y1 = max(0, y1 - pad)
            x2 = min(im_rgb.shape[1], x2 + pad)
            y2 = min(im_rgb.shape[0], y2 + pad)

            plate_img = im_rgb[y1:y2, x1:x2]

            preprocessed = preprocess_plate(plate_img)
            result = reader.readtext(preprocessed, allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
            filtered = [res for res in result if res[2] > 0.4]

            if filtered:
                best = max(filtered, key=lambda x: x[2])
                text = best[1]
                conf = best[2]
                label = f"{text} ({conf:.2f})"
                color = (0, 255, 0)
            else:
                text = "❌ Not confidently detected"
                label = text
                color = (0, 0, 255)

            # Draw rectangle and text on the image
            cv2.rectangle(im_rgb, (x1, y1), (x2, y2), color, 2)
            cv2.putText(im_rgb, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            st.markdown(f"### Plate {i+1}")
            st.image(plate_img, caption="📸 Cropped Plate", width=300)
            st.image(preprocessed, caption="🧪 Preprocessed", width=300)
            st.markdown(f"**Detected Text:** `{text}`")

            all_texts.append(f"Plate {i+1}: {text}")

        # Show final image with all boxes + text
        st.image(im_rgb, caption="🟩 Detection + OCR Result", use_column_width=True)

        # Download button for all text
        if all_texts:
            text_output = "\n".join(all_texts)
            txt_bytes = text_output.encode("utf-8")
            st.download_button(
                label="📥 Download Extracted License Plate Texts",
                data=txt_bytes,
                file_name="license_plates.txt",
                mime="text/plain"
            )
