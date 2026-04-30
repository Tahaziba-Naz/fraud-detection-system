import streamlit as st
import pickle
import numpy as np
import pandas as pd
df = pd.read_csv('../data/creditcard.csv')
df = df.drop(['Time'], axis=1)

model = pickle.load(open('../model/fraud_model.pkl', 'rb'))

st.title("Fraud Detection System")
option = st.selectbox("Select Transaction Type", ["Random", "Fraud Only"])
if st.button("Use Random Transaction"):
    if option == "Fraud Only":
        filtered_df = df[df['Class'] == 1]
    else:
        filtered_df = df
    random_txn = filtered_df.sample(1)
    features = random_txn.drop('Class', axis=1).values
    actual = random_txn['Class'].values[0]
    prediction = model.predict(features)[0]
    prob = model.predict_proba(features)[0][1]
    st.write("Transaction Analysis")
    st.dataframe(random_txn)
    st.write(f"Actual: {'Fraud' if actual==1 else 'Normal'}")
    st.write(f"Predicted: {'Fraud' if prediction==1 else 'Normal'}")
    st.write(f"Fraud Probability: {prob:.2f}")
    if prob > 0.8:
        st.error("HIGH RISK")
    elif prob > 0.5:
        st.warning("MEDIUM RISK")
    else:
        st.success("LOW RISK")
st.write("Enter transaction amount to check fraud risk")

amount = st.number_input("Transaction Amount")

if st.button("Check Fraud"):

    features = np.zeros((1, 29))
    features[0][-1] = amount
    prediction = model.predict(features)[0]
    prob = model.predict_proba(features)[0][1]

    if prediction == 1:
        st.error(f" Fraud Detected! Probability: {prob:.2f}")
    else:
        st.success(f" Safe Transaction. Probability: {prob:.2f}")

    if prob > 0.8:
        st.error(" HIGH RISK")
    elif prob > 0.5:
        st.warning(" MEDIUM RISK")
    else:
        st.success("LOW RISK")