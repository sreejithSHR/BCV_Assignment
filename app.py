import cv2
import numpy as np
import streamlit as st

def apply_basic_operations(img, operation):
    if operation == 'Invert':
        return 255 - img
    elif operation == 'Grayscale':
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif operation == 'Resize':
        dim = (1920, 1080)
        return cv2.resize(img, dim)
    elif operation == 'Rotate':
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    else:
        return img

def apply_edge_detection(img, edge_method):
    if edge_method == 'Canny':
        return cv2.Canny(img, 100, 200)
    elif edge_method == 'LOG':
        # Apply Laplacian of Gaussian (LoG)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        log_result = cv2.Laplacian(cv2.GaussianBlur(gray_img, (5, 5), 0), cv2.CV_64F)
        normalized_log_result = (log_result - log_result.min()) / (log_result.max() - log_result.min())
        return normalized_log_result
    elif edge_method == 'DOG':
        # Apply Difference of Gaussians (DoG)
        blurred_img1 = cv2.GaussianBlur(img, (3, 3), 0)
        blurred_img2 = cv2.GaussianBlur(img, (5, 5), 0)
        return blurred_img1 - blurred_img2
    else:
        return img


def apply_filter(img, filter_type, size):
    if filter_type == 'Gaussian Blur':
        return cv2.GaussianBlur(img, (size, size), 0)
    elif filter_type == 'Median Blur':
        return cv2.medianBlur(img, size)
    elif filter_type == 'Bilateral':
        return cv2.bilateralFilter(img, size, 75, 75)
    else:
        return img

def main():
    st.title('Image Filtering & Operations')

    # Upload image
    image_file = st.file_uploader("Choose and upload an image", type=['jpg', 'jpeg', 'png'])

    if image_file is not None:
        # Load image
        file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)

        # Sidebar
        st.sidebar.title("MENU")
        operation_type = st.sidebar.selectbox('Select Operation Type', ['Basic Operations', 'Edge Detection Techniques'])

        if operation_type == 'Basic Operations':
            st.sidebar.subheader('Basic Operations')
            basic_operation = st.sidebar.radio('Select Operation', ['None', 'Invert', 'Grayscale', 'Resize', 'Rotate'])
            output = apply_basic_operations(img, basic_operation)
        elif operation_type == 'Edge Detection Techniques':
            st.sidebar.subheader('Edge Detection Techniques')
            edge_method = st.sidebar.radio('Select Edge Detection Method', ['None', 'Canny', 'LOG', 'DOG'])
            output = apply_edge_detection(img, edge_method)
        else:
            output = img

        st.subheader('Filters')
        filter_type = st.radio('Select Filter', ['None', 'Gaussian Blur', 'Median Blur', 'Bilateral'])
        filter_size = st.slider('Filter Size', 3, 11, 5, 2)
        filtered_img = apply_filter(img, filter_type, filter_size)

        # Display images
        st.image([img, filtered_img, output], width=300, caption=['Original Image', 'Filtered Image', 'Processed Image'])

if __name__ == '__main__':
    main()
