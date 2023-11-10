import streamlit as st
import cv2
from sklearn.cluster import KMeans
import numpy as np

def load_image(file):
    file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    return img

def calculate_luminance(color):
    return 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = load_image(uploaded_file)
    height, width, _ = img.shape

    # Reshape and cluster
    pixels = img.reshape(-1, 3)
    num_colors = 5
    kmeans = KMeans(n_clusters=num_colors)
    kmeans.fit(pixels)
    colors = kmeans.cluster_centers_.astype(int)

    # Sort colors by luminance
    colors = sorted(colors, key=calculate_luminance)

    # Circle drawing
    circle_radius = min(height, width) // (2 * num_colors)
    if width >= height:  # Wider image
        spacing = width // (num_colors + 1)
        for i, color in enumerate(colors):
            color_tuple = tuple(map(int, color))
            x_position = (i + 1) * spacing
            cv2.circle(img, (x_position, height // 2), circle_radius, color_tuple, -1)
    else:  # Taller image
        spacing = height // (num_colors + 1)
        for i, color in enumerate(colors):
            color_tuple = tuple(map(int, color))
            y_position = (i + 1) * spacing
            cv2.circle(img, (width // 2, y_position), circle_radius, color_tuple, -1)

    # Display
    img_display = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    st.subheader("Color Paletter")
    st.image(img_display, use_column_width=True)
