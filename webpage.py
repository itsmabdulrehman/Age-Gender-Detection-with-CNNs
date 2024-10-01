import streamlit as st
from detect import detect_image, detect_video

def main():
    st.title("Face Analysis App")

    # Image Detection Section
    st.header("Image Detection")
    st.write("Upload an image, and the app will detect faces in the image and provide analysis.")
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Call detect_image with the uploaded file object
        result_img = detect_image(uploaded_file)
        st.image(result_img, caption="Result", use_column_width=True)

    # Webcam Detection Section
    st.header("Webcam Detection")
    st.write("Click the 'Start Webcam' button to enable your webcam. The app will detect faces in real-time.")
    if st.button("Start Webcam"):
        detect_video(0)

if __name__ == "__main__":
    main()
