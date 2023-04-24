import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import shap

st.title("Title")
st.subheader("Subtitle")

df = pd.read_csv("../df.csv")
client = st.sidebar.slider("Select a client number", 0, len(df)-1, 0)
client = df.iloc[client]
st.write("Welcome, ", client.SK_ID_CURR)

model = joblib.load("../modelSb")

shap_values = joblib.load("../shapSb")
fig, ax = plt.subplots()
shap.summary_plot(shap_values, features=client, plot_type="bar", show=False)
st.pyplot(fig)