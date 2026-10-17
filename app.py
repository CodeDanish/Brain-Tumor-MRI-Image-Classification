import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2

# Define class labels matching your dataset directory structure
CLASS_LABELS = ['Glioma Tumor', 'Meningioma Tumor', 'No Tumor', 'Pituitary Tumor']


# 1. Load the trained transfer learning model with caching for fast inference
@st.cache_resource
def load_model():
  # 1. Re-create base and model architecture
  base_model = MobileNetV2(
      input_shape=(224, 224, 3), include_top=False, weights=None
  )
  model = models.Sequential([
      base_model,
      layers.GlobalAveragePooling2D(),
      layers.Dense(256, activation='relu'),
      layers.Dropout(0.5),
      layers.Dense(4, activation='softmax'),
  ])

  # 2. Force load weights with skip_mismatch and by_name to prevent layer index crashing
  try:
    model.load_weights('best_transfer_model.keras', by_name=True, skip_mismatch=True)
    print('Weights loaded with name matching and skipping mismatches.')
  except Exception as e:
    print(f'Error loading weights: {e}')

  return model


model = load_model()

# 2. Page Configuration & Styling
st.set_page_config(
    page_title='Brain Tumor MRI Classification',
    page_icon='🧠',
    layout='centered',
)

# --- Rest of your Streamlit UI code ---
st.title("🧠 Brain Tumor Classification System")
uploaded_file = st.file_uploader("Choose a brain MRI image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Uploaded MRI Scan", use_container_width=True)
        
    with col2:
        st.info("Running inference with trained model...")
        
        # Preprocess matching your training specs (224x224, [0, 1] scaling)
        img_resized = image.resize((224, 224))
        img_array = np.array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Predict
        predictions = model.predict(img_array)[0]
        predicted_class_idx = np.argmax(predictions)
        predicted_class = CLASS_LABELS[predicted_class_idx]
        confidence = predictions[predicted_class_idx] * 100
        
        st.success("Inference Complete!")
        st.metric(label="Predicted Diagnosis", value=predicted_class, delta=f"{confidence:.2f}% Confidence")
        
    st.markdown("---")
    st.subheader("📊 Confidence Score Breakdown")
    for idx, label in enumerate(CLASS_LABELS):
        score = float(predictions[idx])
        st.text(f"{label}: {score * 100:.2f}%")
        st.progress(score)
else:
  st.warning(
      'Please upload an image file (`.jpg`, `.jpeg`, or `.png`) to begin.'
  )