import io
from pathlib import Path

import streamlit as st
from PIL import Image
from ultralytics import YOLO


st.set_page_config(page_title="YOLOv8 Inference - Humans/Cars", layout="wide")

st.title("YOLOv8 Inference - Humans & Cars")

model_path = st.text_input("D:\Projects\Human Movement Detection", value="yolov8x.pt")
conf = st.slider("Confidence", 0.0, 1.0, 0.25, 0.01)
iou = st.slider("IoU", 0.0, 1.0, 0.45, 0.01)
imgsz = st.select_slider("Image size", options=[320, 416, 512, 640, 800, 960, 1280], value=640)
device = st.selectbox("Device", ["0", "cpu"], index=0)

uploaded = st.file_uploader("Upload image", type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"])

if uploaded is not None:
    try:
        img = Image.open(io.BytesIO(uploaded.read())).convert("RGB")
    except Exception as e:
        st.error(f"Could not read image: {e}")
        st.stop()

    st.image(img, caption="Original", use_column_width=True)

    if st.button("Run Inference"):
        model_file = Path(model_path)
        if not model_file.exists():
            st.error(f"Model not found: {model_path}")
            st.stop()

        model = YOLO(str(model_file))
        # COCO: person=0, car=2
        results = model.predict(
            source=img,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            device=device,
            classes=[0, 2],
            verbose=False,
        )

        if not results:
            st.warning("No results returned.")
            st.stop()

        res = results[0]
        plotted = res.plot()
        st.image(plotted, caption="Prediction", use_column_width=True)
        if res.boxes is not None and res.boxes.cls is not None:
            st.write(f"Detections: {len(res.boxes.cls)}")
