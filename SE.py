import io
import matplotlib.pyplot as plt
import seaborn as sns
from pycaret.regression import *
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 600
from pycaret.regression import load_model, predict_model,setup
import streamlit as st
import pandas as pd
import numpy as np
import shap
import io
import joblib
import matplotlib.pyplot as plt
import plotly.express as px

model=load_model("specific-energy")

def predict(model, df2):
    predictions_df = predict_model(estimator=model, data=df2)
    predictions = predictions_df['SE (MJ/m^3)'][0]
    return predictions

def run():

    #from PIL import Image
    #image = Image.open('logo.png')
    #image_hospital = Image.open('hospital.jpg')

    #st.image(image,use_column_width=False)

    add_selectbox = st.sidebar.selectbox(
    "How would you like to predict?",
    ("Online", "Batch"))

    st.sidebar.info('This app is created to classify a soft ground tunnel lithology')
    #st.sidebar.success('https://www.pycaret.org')
    
    #st.sidebar.image(image_hospital)

    #st.title("Insurance Charges Prediction App")

    if add_selectbox == 'Online':
      pressure_gauge1 = st.number_input('Pressure gauge 1 (kPa)', min_value=0, value=0)
      pressure_gauge2 = st.number_input('Pressure gauge 2 (kPa)', min_value=0, value=0)
      pressure_gauge3 = st.number_input('Pressure gauge 3 (kPa)', min_value=0, value=0)
      pressure_gauge4 = st.number_input('Pressure gauge 4 (kPa)', min_value=0, value=0)
      digging_velocity_left = st.number_input('Digging velocity left (mm/min)', min_value=0, value=0)
      digging_velocity_right = st.number_input('Digging velocity right (mm/min)', min_value=0, value=0)
      shield_jack_stroke_left = st.number_input('Shield jack stroke left (mm)', min_value=0, value=0)
      shield_jack_stroke_right = st.number_input('Shield jack stroke right (mm)', min_value=0, value=0)
      propulsion_pressure = st.number_input('Propulsion pressure (MPa)', min_value=0, value=0)
      total_thrust = st.number_input('Total thrust (kN)', min_value=0, value=0)
      cutter_torque = st.number_input('Cutter torque (kNm)', min_value=0, value=0)
      cutterhead_rotation_speed = st.number_input('Cutterhead rotation speed (rpm)', min_value=0, value=0)
      screw_pressure = st.number_input('Screw pressure (MPa)', min_value=0, value=0)
      screw_rotation_speed = st.number_input('Screw rotation speed (rpm)', min_value=0, value=0)
      gate_opening = st.number_input('Gate opening (%)', min_value=0, max_value=100, value=0)
      mud_injection_pressure = st.number_input('Mud injection pressure (MPa)', min_value=0, value=0)
      add_mud_flow = st.number_input('Add mud flow (L/min)', min_value=0, value=0)
      back_in_injection_rate = st.number_input('Back in injection rate (%)', min_value=0, max_value=100, value=0)
        
      output = ""


      input_dict = {'pressure_gauge1' : pressure_gauge1, 'pressure_gauge2' : pressure_gauge2, 'pressure_gauge3' : pressure_gauge3, 'pressure_gauge4' : pressure_gauge4, 'digging_velocity_left' : digging_velocity_left, 
      'digging_velocity_right' : digging_velocity_right,'shield_jack_stroke_left' : shield_jack_stroke_left,'shield_jack_stroke_right' : shield_jack_stroke_right,
      'propulsion_pressure' : propulsion_pressure,'total_thrust' : total_thrust,'cutter_torque' : cutter_torque,'cutterhead_rotation_speed' : cutterhead_rotation_speed,
      'screw_pressure' : screw_pressure,'screw_rotation_speed' : screw_rotation_speed,'gate_opening' : gate_opening,'mud_injection_pressure' : mud_injection_pressure,'add_mud_flow' : add_mud_flow,
      'back_in_injection_rate':back_in_injection_rate}
      
      input_df = pd.DataFrame([input_dict])

        if st.button("Predict"):
            output = predict(model=model, input_df=df2)
            output = '$' + str(output)

        st.success('The output is {}'.format(output))

    if add_selectbox == 'Batch':

        file_upload = st.file_uploader("Upload csv file for predictions", type=["csv","xlsx"])

        if file_upload is not None:
            if file_upload.type == 'application/vnd.ms-excel':  # Check if the uploaded file is in Excel format
                data = pd.read_excel(file_upload)
            else:
                data = pd.read_csv(file_upload)
            
            data = data.dropna()
            predictions = predict_model(estimator=model,data=data)
            st.write(predictions)
          

if __name__ == '__main__':
    run()
