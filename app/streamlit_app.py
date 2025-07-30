import streamlit as st
import pandas as pd
import xgboost as xgb

bst = xgb.Booster()
bst.load_model('model.bst')

st.title("Prévision pluie pour demain 🌧️")
st.write("Entrez vos mesures météo d'aujourd'hui :")

precip_lag1 = st.number_input("Précipitations hier (mm)", min_value=0.0, step=0.1)
press_lag1 = st.number_input("Pression moyenne hier (hPa)", value=1013.0, step=0.1)
precip_3d_sum = st.number_input("Somme précipitations (3 derniers jours) (mm)", step=0.1)
temp_lag1 = st.number_input("Température moyenne hier (°C)", step=0.1)

if st.button("Prédire pluie demain"):
    data = pd.DataFrame([{
        "precip_lag1": precip_lag1,
        "press_lag1": press_lag1,
        "precip_3d_sum": precip_3d_sum,
        "temp_lag1": temp_lag1
    }])
    dmat = xgb.DMatrix(data)
    prob = bst.predict(dmat)[0]
    st.success(f"🌧️ Probabilité de pluie demain : {prob:.2%}")
