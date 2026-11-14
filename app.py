
import streamlit as st
import cv2
import numpy as np
import joblib
from skimage.feature import hog
from PIL import Image

# -----------------------------------
# 1. Load the trained model and scaler
# -----------------------------------

model = joblib.load("face_mask_svm.pkl")
scaler = joblib.load("face_mask_scaler.pkl")


# -----------------------------------
# 2. Load Haar Cascade face detector
# -----------------------------------

face_cascade = cv2.CascadeClassifier(
    "haarcascade_frontalface_default.xml"
)

if face_cascade.empty():
    st.error("Error: Haar Cascade file could not be loaded.")
    st.stop()


# -----------------------------------
# 3. HOG feature extraction
# -----------------------------------

def extract_hog_features(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    resized = cv2.resize(gray, (64, 64))

    features = hog(
        resized,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys"
    )

    return features


# -----------------------------------
# 4. Streamlit page configuration
# -----------------------------------

st.set_page_config(
    page_title="Face Mask Detection ",
    page_icon="😷",
    layout="centered"
)

st.title("😷 Face Mask Detection System Using SVM ")

st.write(
    "Upload an image to detect faces and identify "
    "whether they are wearing masks."
)


# -----------------------------------
# 5. Upload image
# -----------------------------------

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)


# -----------------------------------
# 6. Process uploaded image
# -----------------------------------

if uploaded_file is not None:

    # Read uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    image = np.array(image)

    # Convert RGB to BGR for OpenCV
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Convert image to grayscale
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # Improve image contrast
    gray = cv2.equalizeHist(gray)

    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=6,
        minSize=(80, 80)
    )

    # Store predictions
    predictions = []

    st.subheader("Detection Results")

    if len(faces) == 0:

        st.warning("No faces detected in the image.")

        st.image(
            image,
            caption="Uploaded Image",
            use_container_width=True
        )

    else:

        # Process each detected face
        for (x, y, w, h) in faces:

            # Crop the face
            face = image_bgr[y:y+h, x:x+w]

            # Extract HOG features
            features = extract_hog_features(face)

            # Reshape features
            features = features.reshape(1, -1)

            # Scale features
            scaled_features = scaler.transform(features)

            # Predict using SVM model
            prediction = model.predict(scaled_features)[0]

            # Identify prediction label
            if prediction == 1:

                label = "With Mask"
                box_color = (0, 180, 0)

            else:

                label = "Without Mask"
                box_color = (0, 0, 255)

            # Store prediction
            predictions.append(label)

            # Draw rectangle around detected face
            cv2.rectangle(
                image_bgr,
                (x, y),
                (x+w, y+h),
                box_color,
                2
            )

            # Display prediction on bounding box
            cv2.putText(
                image_bgr,
                label,
                (x, max(y-10, 25)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                box_color,
                2
            )

        # Convert BGR back to RGB
        result_image = cv2.cvtColor(
            image_bgr,
            cv2.COLOR_BGR2RGB
        )

        # Display processed image
        st.image(
            result_image,
            caption="Face Mask Detection Output",
            use_container_width=True
        )

        # Display number of detected faces
        st.success(f"Detected {len(faces)} face(s).")

        # -----------------------------------
        # 7. Display prediction results below
        # -----------------------------------

        st.subheader("Prediction Results")

        for i, prediction in enumerate(predictions):

            if prediction == "With Mask":

                st.success(
                    f"Face {i+1} - Prediction: With Mask"
                )

            else:

                st.error(
                    f"Face {i+1} - Prediction: Without Mask"
                )


# -----------------------------------
# 8. Footer
# -----------------------------------

st.markdown("---")

st.caption("Machine Learning Project | HOG + SVM")
