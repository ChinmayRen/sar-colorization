import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from PIL import Image
import io
from tensorflow.keras import layers
import streamlit.components.v1 as components


# Define the SelfAttention class that was used in your models
class SelfAttention(layers.Layer):
    def __init__(self, filters, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.gamma = self.add_weight(shape=(), initializer='zeros', trainable=True)
        
        # Convolution layers for query, key, value
        self.query_conv = layers.Conv2D(filters // 8, 1, padding='same')
        self.key_conv = layers.Conv2D(filters // 8, 1, padding='same')
        self.value_conv = layers.Conv2D(filters, 1, padding='same')

    def call(self, x):
        batch_size = tf.shape(x)[0]
        h = tf.shape(x)[1]
        w = tf.shape(x)[2]
        
        # Project inputs to query, key, value
        query = self.query_conv(x)  # [B, H, W, C//8]
        key = self.key_conv(x)      # [B, H, W, C//8]
        value = self.value_conv(x)  # [B, H, W, C]
        
        # Reshape for attention computation
        query_flat = tf.reshape(query, [batch_size, h * w, -1])  # [B, H*W, C//8]
        key_flat = tf.reshape(key, [batch_size, h * w, -1])      # [B, H*W, C//8]
        
        # Attention matrix (H*W x H*W)
        attention = tf.matmul(query_flat, key_flat, transpose_b=True)  # [B, H*W, H*W]
        attention = tf.nn.softmax(attention, axis=-1)
        
        # Apply attention to value
        value_flat = tf.reshape(value, [batch_size, h * w, -1])  # [B, H*W, C]
        out_flat = tf.matmul(attention, value_flat)  # [B, H*W, C]
        out = tf.reshape(out_flat, [batch_size, h, w, self.filters])
        
        # Residual connection
        return self.gamma * out + x

    def get_config(self):
        config = super().get_config()
        config.update({'filters': self.filters})
        return config

# Function to preprocess L channel
def preprocess_l_channel(image):
    # Resize to 256x256
    image = cv2.resize(image, (256, 256))
    
    # Convert to LAB
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2Lab)
    
    # Extract L channel
    l_channel = lab[:, :, 0]
    
    # Normalize to [-1, 1]
    l_channel = (l_channel.astype(np.float32) / 127.5) - 1.0
    
    return np.expand_dims(l_channel, axis=-1)

# Function to denormalize images for display
def denormalize(image):
    return (image * 0.5 + 0.5).clip(0, 1)

# Main Streamlit app
def main():
    st.title("Terrain-Specific SAR Colorization")
    
    st.sidebar.header("Model Selection")
    model_options = ["Agricultural", "Barren", "Urban", "Grassland"]
    model_choice = st.sidebar.selectbox("Choose Terrain Type:", 
                                       options=model_options)
    
    # Map selection to model number
    model_map = {
        "Agricultural": 1,
        "Barren": 2,
        "Urban": 3,
        "Grassland": 4
    }
    
    model_no = model_map[model_choice]
    
    # File uploader for input image
    st.header("Upload SAR Image")
    uploaded_file = st.file_uploader("Choose a SAR image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Process the image when the user clicks the button
        if st.button("Generate Colorized Image"):
            st.write("Processing...")
            
            # Convert PIL image to OpenCV format
            img_array = np.array(image)
            if len(img_array.shape) == 2:  # If grayscale, convert to RGB
                img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
            elif img_array.shape[2] == 4:  # If RGBA, convert to RGB
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
            
            # Try to load model
            try:
                # Path to model based on user selection
                model_paths = {
                    1: "models/lab_agri_generator_epoch_100.h5",
                    2: "models/lab_barren_generator_epoch_100.h5",
                    3: "models/lab_urban_generator_epoch_100.h5",
                    4: "models/lab_grassland_generator_epoch_100.h5"
                }
                
                model_path = model_paths[model_no]
                
                if not os.path.exists(model_path):
                    st.error(f"Model not found at {model_path}. Please ensure models are in the 'models' directory.")
                    return
                
                # Load the selected model
                with st.spinner(f"Loading {model_choice} model..."):
                    generator = tf.keras.models.load_model(
                        model_path, 
                        custom_objects={'SelfAttention': SelfAttention}
                    )
                
                # Preprocess the image
                with st.spinner("Preprocessing image..."):
                    l_channel = preprocess_l_channel(img_array)
                    
                # Generate colorized image
                with st.spinner("Generating colorized image..."):
                    generated_rgb = generator.predict(np.expand_dims(l_channel, axis=0))[0]
                    
                # Prepare images for display
                l_display = denormalize(l_channel.squeeze())
                generated_display = denormalize(generated_rgb)
                
                # Create figure for display
                fig, ax = plt.subplots(1, 2, figsize=(12, 6))
                
                # Display input L channel
                ax[0].imshow(l_display, cmap='gray')
                ax[0].set_title('Input L Channel')
                ax[0].axis('off')
                
                # Display generated color image
                ax[1].imshow(generated_display)
                ax[1].set_title('Generated Color Image')
                ax[1].axis('off')
                
                plt.tight_layout()
                
                # Display in Streamlit
                st.pyplot(fig)
                
                # Save generated image
                generated_pil = Image.fromarray((generated_display * 255).astype(np.uint8))
                
                # Create a BytesIO object
                buf = io.BytesIO()
                generated_pil.save(buf, format="PNG")
                
                # Provide download button
                st.download_button(
                    label="Download Colorized Image",
                    data=buf.getvalue(),
                    file_name=f"colorized_{model_choice}.png",
                    mime="image/png"
                )
                
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")
    
    # Instructions
    st.sidebar.markdown("## Instructions")
    st.sidebar.markdown("""
    1. Select a terrain type from the dropdown menu
    2. Upload a SAR image (grayscale or RGB)
    3. Click 'Generate Colorized Image'
    4. Download the result using the button below the generated image
    """)
    
    # About section
    st.sidebar.markdown("## About")
    st.sidebar.markdown("""
    This app uses deep learning models to colorize SAR (Synthetic Aperture Radar) 
    images based on different terrain types. Each model is specialized for a 
    specific terrain type and produces more accurate colors for that terrain.
    """)

if __name__ == "__main__":
    main()
