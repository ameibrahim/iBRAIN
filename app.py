import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Dictionary for model file paths
MODEL_PATHS = {
    "MobileNet": ["models/MobileNet_mrinomri.keras", "models/MobileNet_tumornotumor.keras"],
    "VGG16": ["models/VGG16_mrinomri.keras", "models/VGG16_tumornotumor.keras"],
    "ResNet50": ["models/ResNet50_mrinomri.keras", "models/ResNet50_tumornotumor.keras"],
    "InceptionV3": ["models/InceptionV3_mrinomri.keras", "models/InceptionV3_tumornotumor.keras"]
}

SIZES = {
    "MobileNet": 224,  # Example, replace with actual layer
    "VGG16": 224,  # Example, replace with actual layer
    "ResNet50": 299,  # Example, replace with actual layer
    "InceptionV3": 299  # Example, replace with actual layer
}

# Placeholder dictionary for last convolutional layer names (to be filled by the user)
LAST_CONV_LAYERS = {
    "MobileNet": "conv_pw_13",  # Example, replace with actual layer
    "VGG16": "block5_conv3",  # Example, replace with actual layer
    "ResNet50": "conv5_block3_3_conv",  # Example, replace with actual layer
    "InceptionV3": "conv2d_187"  # Example, replace with actual layer
}

def grad_cam(img_array, model, layer_name, pred_index=None):
    # Get last convolutional layer dynamically
    last_conv_layer = model.layers[0].get_layer(layer_name)  # ✅ Corrected


    # Define the Grad-CAM model
    grad_model = tf.keras.models.Model(
        inputs=model.layers[0].input, 
        outputs=[last_conv_layer.output, model.layers[0].output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])  # Get the predicted class index
            loss = tf.gather(predictions, pred_index, axis=1)  # ✅ Fix tensor indexing


    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_mean(conv_outputs * pooled_grads, axis=-1)
    heatmap = np.maximum(heatmap, 0)  # Apply ReLU
    heatmap /= np.max(heatmap)  # Normalize
    return heatmap

# Function to preprocess image
def preprocess_image(uploaded_file):
    img = image.load_img(uploaded_file, target_size=(SIZES[selected_model], SIZES[selected_model]))  # Resize for model
    img_array = image.img_to_array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array, img

# Streamlit UI
st.title("MRI Tumor Classification with Grad-CAM")

# Dropdown to select the model
selected_model = st.selectbox("Select a Model Architecture", list(MODEL_PATHS.keys()))

# Check if a model is selected
if selected_model:
    # Load models dynamically
    mri_model_path, tumor_model_path = MODEL_PATHS[selected_model]
    mri_model = load_model(mri_model_path)
    tumor_model = load_model(tumor_model_path)
    last_conv_layer = LAST_CONV_LAYERS[selected_model]  # Get correct last conv layer

    # File upload section
    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
        
        img_array, original_img = preprocess_image(uploaded_file)

        # Step 1: Check if Image is MRI
        is_mri = mri_model.predict(img_array)[0][0]

        if is_mri > 0.5:
            st.success(f"✅ The uploaded image is classified as an MRI using {selected_model}.")
            
            # Step 2: Check if MRI has a tumor
            has_tumor = tumor_model.predict(img_array)[0][0]

            if has_tumor > 0.5:
                st.error("🔴 The MRI indicates the presence of a tumor.")
            else:
                st.success("🟢 No tumor detected in the MRI.")

            # Step 3: Generate Grad-CAM for visualization
            st.subheader("Grad-CAM Heatmaps")

            mri_heatmap = grad_cam(img_array, mri_model, last_conv_layer)
            tumor_heatmap = grad_cam(img_array, tumor_model, last_conv_layer)

            def apply_heatmap(heatmap, original_img):
                heatmap = cv2.resize(heatmap, (original_img.size[0], original_img.size[1]))
                heatmap = np.uint8(255 * heatmap)
                heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                superimposed_img = cv2.addWeighted(np.array(original_img), 0.6, heatmap, 0.4, 0)
                return superimposed_img

            mri_cam = apply_heatmap(mri_heatmap, original_img)
            tumor_cam = apply_heatmap(tumor_heatmap, original_img)

            col1, col2 = st.columns(2)
            with col1:
                st.image(mri_cam, caption=f"{selected_model} MRI Detection Grad-CAM", use_container_width=True)
            with col2:
                st.image(tumor_cam, caption=f"{selected_model} Tumor Detection Grad-CAM", use_container_width=True)

        else:
            st.warning(f"⚠️ The uploaded image is classified as NOT an MRI using {selected_model}.")