import streamlit as st
import requests

from PIL import Image
st.set_page_config(
    page_title="MNIST Digit Recognition",
    page_icon="🔢",
    layout="centered"
)


st.title("🔢 MNIST Digit Recognition")

st.write(
    "Upload a handwritten digit image "
    "and our ANN model will predict the digit."
)

API_URL = "http://127.0.0.1:8000/predict"

uploaded_file = st.file_uploader(
    "Upload an image of a handwritten digit",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded Image",
        width=250
    )


    if st.button("🔍 Predict Digit"):

        try:

            with st.spinner("Predicting..."):

                uploaded_file.seek(0)

                response = requests.post(
                    API_URL,
                    files={
                        "file": (
                            uploaded_file.name,
                            uploaded_file,
                            uploaded_file.type
                        )
                    }
                )


            if response.status_code == 200:

                result = response.json()

                predicted_digit = result["prediction"]

                confidence = result["confidence"]

                st.success(
                    f"Predicted Digit: {predicted_digit}"
                )

                st.info(
                    f"Confidence: {confidence * 100:.2f}%"
                )


            else:

                st.error(
                    "API returned an error."
                )


        except requests.exceptions.ConnectionError:

            st.error(
                "Could not connect to FastAPI. "
                "Please make sure the FastAPI server is running."
            )


        except Exception as e:

            st.error(
                f"An error occurred: {e}"
            )