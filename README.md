
import os
os.environ["KERAS_BACKEND"] = "tensorflow"

import streamlit as st
import numpy as np
import requests
import json
from PIL import Image
from huggingface_hub import hf_hub_download
import keras

st.set_page_config(page_title="Cloud & Rain Predictor", page_icon="☁️")

st.title("☁️ Cloud & Rain Prediction")
st.write("Upload a sky photo to identify the cloud type and check local weather forecasts.")

@st.cache_resource
def load_cloud_model():
    model = keras.saving.load_model(
        "hf://mubaraknumann/genera-cloud-image-classification"
    )
    label_file = hf_hub_download(
        repo_id="mubaraknumann/genera-cloud-image-classification",
        filename="label_mapping.json"
    )
    with open(label_file, "r") as f:
        labels = json.load(f)["int_to_label"]
    return model, {int(k): v for k, v in labels.items()}

photo = st.file_uploader(
    "Upload a cloud photo",
    type=["jpg", "jpeg", "png"]
)

if photo:
    image = Image.open(photo).convert("RGB")
    st.image(image, caption="Your uploaded sky photo", use_container_width=True)

    if st.button("Identify Cloud Type"):
        try:
            with st.spinner("AI is analysing the image..."):
                model, labels = load_cloud_model()
                image = image.resize((299, 299))
                array = np.array(image, dtype=np.float32) / 255.0
                prediction = model.predict(np.expand_dims(array, axis=0), verbose=0)[0]
                class_id = int(np.argmax(prediction))

            st.success(f"Predicted cloud type: {labels[class_id]}")
            st.write(f"Model confidence: {float(prediction[class_id]) * 100:.1f}%")
            st.info(
                "This is an AI image classification result, not a guarantee of rain."
            )
        except Exception as error:
            st.error("The AI model could not load or predict.")
            st.write("Please share this error screenshot so we can fix it.")
            st.code(str(error)[:1200])

st.divider()
st.subheader("🌧️ Local Weather Forecast")

city = st.text_input("Enter your city", value="Vijayawada")

if st.button("Check Rain Forecast"):
    try:
        geo = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": city, "count": 1, "language": "en", "format": "json"},
            timeout=20
        )
        geo.raise_for_status()
        places = geo.json().get("results", [])

        if not places:
            st.warning("City not found. Check the spelling and try again.")
        else:
            place = places[0]
            weather = requests.get(
                "https://api.open-meteo.com/v1/forecast",
                params={
                    "latitude": place["latitude"],
                    "longitude": place["longitude"],
                    "hourly": "precipitation_probability,precipitation",
                    "forecast_days": 1,
                    "timezone": "auto"
                },
                timeout=20
            )
            weather.raise_for_status()
            data = weather.json()["hourly"]
            chances = [
                x for x in data.get("precipitation_probability", [])
                if x is not None
            ]

            st.write(f"Location: {place['name']}, {place.get('country', '')}")
            if chances:
                st.metric("Highest rain probability in today's hourly forecast",
                          f"{max(chances)}%")
                st.caption(
                    "This is a weather forecast for the selected location, "
                    "not a prediction derived from the uploaded photo."
                )
            else:
                st.warning("Rain probability data is unavailable.")
    except Exception as error:
        st.error("Weather data could not be loaded. Check your internet connection.")
        st.code(str(error)[:800])

