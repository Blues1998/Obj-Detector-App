import streamlit as st
import cv2
from PIL import Image
import numpy as np
import os


CLASSES = [
    "background",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
PROTOTXT = 'model/MobileNetSSD_deploy.prototxt.txt'
MODEL = 'model/MobileNetSSD_deploy.caffemodel'


@st.cache
def detect(our_image):
    blob = cv2.dnn.blobFromImage(
        cv2.resize(our_image, (300, 300)), 0.007843, (300, 300), 127.5
    )
    net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)
    net.setInput(blob)
    detections = net.forward()
    return detections


@st.cache
def annotate_image(image, detections, confidence_threshold=0.5):
    (h, w) = image.shape[:2]
    labels = []
    for i in np.arange(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > confidence_threshold:
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # display the prediction
            label = f"{CLASSES[idx]}: {round(confidence * 100, 2)}%"
            labels.append(label)
            cv2.rectangle(image, (startX, startY), (endX, endY), COLORS[idx], 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(
                image, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLORS[idx], 2
            )
        return image, labels


def main():
    st.title("Object Detection App!!!")
    choice = st.sidebar.selectbox("Pick what to detect", ["Objects"])

    if choice == "Objects":
        st.subheader("We will be detecting faces today !! Yayy!!")

        image_file = st.file_uploader("Upload your image", type=["jpeg", "jpg", "png"])
        confidence_threshold = st.slider(
            "Confidence threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.05
        )

        # If the image file exists then open it as np array
        if image_file is not None:
            image = np.array(Image.open(image_file))
            if st.button("Process"):
                detections = detect(image)
                image, labels = annotate_image(image, detections, confidence_threshold)
                st.image(image, caption=f"Processed Image", use_column_width=True,)


if __name__ == '__main__':
    main()
