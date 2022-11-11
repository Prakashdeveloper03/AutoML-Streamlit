import os
import pandas as pd
import streamlit as st
from streamlit_pandas_profiling import st_profile_report
from pycaret.regression import setup, compare_models, pull, save_model

# setting app's title, icon & layout
st.set_page_config(page_title="AutoML", page_icon="⚡")

# sample dataset
if os.path.exists("./data/dataset.csv"):
    df = pd.read_csv("./data/dataset.csv", index_col=None)

with st.sidebar:
    st.image("./imgs/inno2.png")
    st.title("AutoML")
    choice = st.selectbox(
        "Navigation", ["Upload", "Profiling", "Modelling", "Download"]
    )
    st.info("This project application helps you build and explore your data.")

match choice:
    case "Upload":
        st.title("Upload Your Dataset")
        if file := st.file_uploader("Upload Your Dataset"):
            df = pd.read_csv(file, index_col=None)
            df.to_csv("data/dataset.csv", index=None)
            st.dataframe(df)

    case "Profiling":
        st.title("Exploratory Data Analysis")
        profile_df = df.profile_report()
        st_profile_report(profile_df)

    case "Modelling":
        chosen_target = st.selectbox("Choose the Target Column", df.columns)
        if st.button("Run Modelling"):
            setup(df, target=chosen_target, silent=True)
            setup_df = pull()
            st.dataframe(setup_df)
            best_model = compare_models()
            compare_df = pull()
            st.dataframe(compare_df)
            save_model(best_model, "model/best_model")

    case "Download":
        with open("model/best_model.pkl", "rb") as f:
            st.download_button("Download Model", f, file_name="model/best_model.pkl")
