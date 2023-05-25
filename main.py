import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import shap
import lightgbm
import sklearn
import imblearn
import numpy as np

st.write("joblib_version:", joblib.__version__)
st.write("streamlit_version:", st.__version__)
st.write("pd_version:", pd.__version__)
st.write("shap_version:", shap.__version__)
st.write("lightgbm_version:", lightgbm.__version__)
st.write("sklearn_version:", sklearn.__version__)

st.title("Loan attribution tool")
st.subheader("Select your client id.")

# Get the user's information
df = pd.read_csv("guys.csv")
client = st.sidebar.slider("Select a client number", 0, len(df)-1, 0)
client = df.iloc[client]
client = client.SK_ID_CURR
client = df[df.SK_ID_CURR == client]
st.write("Welcome, client id", client.SK_ID_CURR.values[0])

# Propose modifications for the client's profile
# (Only the major features)
major_features = joblib.load("major_features")
client_bis = client.copy()
for i in major_features.name:
    client_bis[i] = st.number_input(i, value=client[i].values[0])

@st.cache_resource
def get_model():
    return joblib.load("modelSb")

# Prediction
if st.button("Predict"):
    model = get_model()
    result = model.predict(client_bis)[0]
    if result == 1:
        result = "Your loan was refused."
    elif result == 0:
        result = "Your loan was granted."
    else:
        result = "Error during prediction."
    client_score = np.round(model.predict_proba(client_bis)[0][0], 2)
    st.write("Your score is " + str(client_score))
    st.write(result)

@st.cache_data
def get_avg():
    dff = pd.read_csv("df.csv")
    #client_shap = dff[dff.SK_ID_CURR == client]
    # #st.write(client[major_features.name])
    df_mean = dff[major_features.name].mean()
    df_mean.columns = ["Feature", "Average_Population"]
    return df_mean
st.write("Characteristics of the average population:")
st.write(get_avg())

shap_values = joblib.load("shapSb5")
#shap_explainer = joblib.load("shap_xpl")
#feature_names = joblib.load("feature_names")
fig, ax = plt.subplots()
#shap.plots.waterfall(shap_explainer, max_display=8)
shap.summary_plot(
    shap_values,
    features=client_bis,
    #feature_names=feature_names,
    max_display=8,
    plot_type="bar",
    show=False,
)
st.pyplot(fig)