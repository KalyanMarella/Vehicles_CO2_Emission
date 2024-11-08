import streamlit as st
import pandas as pd
from steps.data_preprocess import category_handling_test,feature_transform_test,feature_scaling_test
import requests

st.title("Vehicle CO2 Emission")

st.write("""
##### This ML-based app provides the Prediction of CO2 emission of vehicles based on the vehicle parameters 
""")


fuel_type = st.sidebar.selectbox("Fuel Type of Vehicle", ["Regular Gasoline", "Premium Gasoline", "Ethanol", "Diesel"])
vehicle_class = st.sidebar.selectbox("Vehicle class", ["STATION WAGON - SMALL", "COMPACT", "MID-SIZE", "STATION WAGON - MID-SIZE",
                                                       "SUV - SMALL", "SPECIAL PURPOSE VEHICLE", "SUBCOMPACT", "MINICOMPACT", 
                                                       "FULL-SIZE", "TWO-SEATER", "MINIVAN", "PICKUP TRUCK - SMALL", 
                                                       "PICKUP TRUCK - STANDARD", "SUV - STANDARD", "VAN - CARGO", 
                                                       "VAN - PASSENGER"])
transmission = st.sidebar.selectbox("Transmission", ["AS6", "AS8", "M6", "A6", "A8", "AM7", "A9", "AS7", "AV", "M5", "AS10", 
                                                     "AM6", "AV7", "AV6", "M7", "A5", "AS9", "A4", "AM8", "A7", "AV8", "A10", 
                                                     "AS5", "AV10", "AM5", "AM9", "AS4"])

engine_size = st.sidebar.number_input("Engine Size", min_value=0, max_value=200)
cylinders = st.sidebar.number_input("Cylinders", min_value=0, max_value=200)
fuel_consumption_comb_l_per_100_km = st.sidebar.number_input("Fuel Consumption in Litres per 100 Km", min_value=0.1, max_value=1999.1, step=0.1)
fuel_consumption_comb_mpg = st.sidebar.number_input("Fuel Consumption Miles per Gallon", min_value=0.1, max_value=1999.1, step=0.1)


def print_values():
    data = pd.DataFrame({
        "vehicle_class": [vehicle_class],
        "engine_size_l": [engine_size],
        "cylinders": [cylinders],
        "transmission": [transmission],
        "fuel_type": [fuel_type],
        "fuel_consumption_comb_l_per_100_km": [fuel_consumption_comb_l_per_100_km],
        "fuel_consumption_comb_mpg": [fuel_consumption_comb_mpg]
    })
    return data

import requests
import streamlit as st
import numpy as np

def get_prediction(data):
    url = "http://localhost:3001/predict_ndarray"  # Updated endpoint URL
    try:
        # Convert the DataFrame to a NumPy array and then to a nested list
        data_array = data.values.tolist()  # This gives you the correct format for input
        st.write("Data sent to model:", data_array)  # For debugging

        # Send the array as JSON to the BentoML service
        response = requests.post(url, json=data_array)
        
        # Check for successful response and parse JSON output
        response.raise_for_status()  # Raises an HTTPError for bad responses
        return response.json()  # If successful, return the prediction response
    except requests.exceptions.RequestException as e:
        st.error(f"Error: Could not get prediction from the model.\n{e}")
        return None


if st.sidebar.button("Submit"):
    st.text("\n")
    data = print_values()
    st.write(data)

    # Get the prediction from the BentoML service
    mapp={"Regular Gasoline":"X","Premium Gasoline":"Z","Diesel":"D","Ethanol":"E"}
    data["fuel_type"]=data["fuel_type"].map(mapp)
    st.write(data)
    data_after_encoding=category_handling_test(data)
    data_after_transform=feature_transform_test(data_after_encoding)
    data_after_scaling=feature_scaling_test(data_after_transform)
    prediction = get_prediction(data_after_scaling)
    
    if prediction is not None:
        st.write("The CO2 Emission of the Vehicle with given parameters is:")
        if prediction is not None:
            st.write("The CO2 Emission of the Vehicle with given parameters is:", prediction)