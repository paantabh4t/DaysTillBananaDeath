import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="Banana Freshness AI",
    page_icon="🍌",
    layout="centered"
)

st.title("🍌 Banana Freshness & Shelf-Life Predictor")
st.write("Upload or capture a photo of a banana to estimate how many days it will stay good.")

# 1. Load the TFLite Model
@st.cache_resource
def load_tflite_model():
    interpreter = tf.lite.Interpreter(model_path="banana_model.tflite")
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_tflite_model()

# Get input and output tensor details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Extract input shape expected by the model (e.g. [1, 224, 224, 3])
input_shape = input_details[0]['shape']
target_height = input_shape[1]
target_width = input_shape[2]

# 2. Preprocess Image
def preprocess_image(image: Image.Image):
    if image.mode != "RGB":
        image = image.convert("RGB")
        
    image = image.resize((target_width, target_height))
    img_array = np.array(image, dtype=np.float32)

    # Normalization: match your training (e.g., / 255.0)
    # Check if model expects UINT8 or FLOAT32:
    if input_details[0]['dtype'] == np.float32:
        img_array = img_array / 255.0

    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# 3. Input Mode Selection (Camera or Upload)
input_mode = st.radio("Choose Input Method:", ("📷 Take Live Photo", "📁 Upload Image"), horizontal=True)

uploaded_file = None
if input_mode == "📷 Take Live Photo":
    uploaded_file = st.camera_input("Snap a picture of the banana")
else:
    uploaded_file = st.file_uploader("Upload banana image...", type=["jpg", "jpeg", "png"])

# 4. Run TFLite Inference
if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Banana", use_container_width=True)

    with st.spinner("Analyzing freshness..."):
        input_tensor = preprocess_image(img)
        
        # Set input tensor and invoke
        interpreter.set_tensor(input_details[0]['index'], input_tensor)
        interpreter.invoke()
        
        # Get prediction output
        output_data = interpreter.get_tensor(output_details[0]['index'])

        # Regression: Extract days remaining
        days = float(output_data[0][0])
        days = max(0.0, round(days, 1))

    # 5. Display Results
    st.markdown("---")
    st.subheader("Freshness Assessment")
    st.metric(label="Estimated Shelf Life", value=f"{days} Days Remaining")

    if days >= 4.0:
        st.success("🟢 **Fresh / Under-ripe**: Will stay good for several days. Store at room temperature.")
    elif days >= 2.0:
        st.info("🟡 **Prime Ripe**: Perfect sweetness and texture for eating today!")
    elif days >= 1.0:
        st.warning("🟠 **Very Ripe / Spotty**: High sugar. Eat soon or use for smoothies.")
    else:
        st.error("🟤 **Overripe / Spoiled**: Best used immediately for banana bread or discard.")