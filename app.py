import streamlit as st
import pandas as pd
import requests

st.title("Iris Classifier")

st.sidebar.header("Iris Features")
sepal_length = st.sidebar.slider("sepal length(cm)", min_value=0.0, max_value=10.0, step=0.1)
sepal_width = st.sidebar.slider("sepal width(cm)", min_value=0.0, max_value=10.0, step=0.1)
petal_length = st.sidebar.slider("petal length(cm)", min_value=0.0, max_value=10.0, step=0.1)
petal_width = st.sidebar.slider("petal width(cm)", min_value=0.0, max_value=10.0, step=0.1)

iris = {
    "sepal_length": sepal_length,
    "sepal_width": sepal_width,
    "petal_length": petal_length,
    "petal_width": petal_width
}

targets = ["setona", "versicolor", "virginica"]

if st.sidebar.button("Predict"):
    # show
    st.write("## Input Value")
    iris_df = pd.DataFrame(iris, index=["data"])
    st.write(iris_df)

    # do
    response = requests.post("http://localhost:8000/predict", json=iris)
    prediction = response.json()["prediction"]

    # show
    st.write("## Prediction")
    st.write(prediction)

    # output
    st.write("## Result")
    st.write("このアイリスはきっと", str(targets[int(prediction)]), "です。")