import streamlit as st
import numpy as np
import tensorflow as tf
import pandas as pd
import requests
from streamlit_lottie import st_lottie
from admin import connect_to_mysql, save_data_to_mysql, close_mysql_connection, view_transaction_data, filter_transaction_data,admin_panel

# Function to load Lottie animation from URL
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Title of the app
st.title("Multiperspective Fraud Detection ")

# Sidebar mode selection
mode = st.sidebar.radio("Select Mode", ("User", "Admin"))

# Load Lottie animation
lottie_url = "https://assets8.lottiefiles.com/packages/lf20_yhTqG2.json"
lottie_hello = load_lottieurl(lottie_url)

# Conditionally display Lottie animation based on mode
if mode == "User":
    with st.sidebar:
        st_lottie(lottie_hello, quality='high')


    st.sidebar.title('Users Features Explanation')
    st.sidebar.markdown("**step**: represents a unit of time where 1 step equals 1 hour")
    st.sidebar.markdown("**type**: type of online transaction")
    st.sidebar.markdown('**amount**: the amount of the transaction')
    st.sidebar.markdown('**oldbalanceOrg**: balance before the transaction')
    st.sidebar.markdown('**newbalanceOrig**: balance after the transaction')
    st.sidebar.markdown('**oldbalanceDest**: initial balance of recipient before the transaction')
    st.sidebar.markdown('**newbalanceDest**: the new balance of recipient after the transaction')

    st.header('User Input Features')
    
    

    def user_input_features():
        step = st.number_input('Step', 0, 3)
        type = st.selectbox('Online Transaction Type', ("CASH IN", "CASH OUT", "DEBIT", "PAYMENT", "TRANSFER"))
        amount = st.number_input("Amount of the transaction")
        oldbalanceOrg = st.number_input("Old balance Origin")
        newbalanceOrig = st.number_input("New balance Origin")
        oldbalanceDest = st.number_input("Old Balance Destination")
        newbalanceDest = st.number_input("New Balance Destination")
        data = {'step': step,
                'type': type,
                'amount': amount,
                'oldbalanceOrg': oldbalanceOrg,
                'newbalanceOrig': newbalanceOrig,
                'oldbalanceDest': oldbalanceDest,
                'newbalanceDest': newbalanceDest}
        features = pd.DataFrame(data, index=[0])
        return features

    input_df = user_input_features()

    # Combines user input features with sample dataset
    # This will be useful for the encoding phase
    fraud_raw = pd.read_csv('samp_online.csv')
    fraud = fraud_raw.drop(columns=['isFraud','nameOrig','nameDest','isFlaggedFraud'])
    df = pd.concat([input_df,fraud],axis=0)

    # Encoding of ordinal features
    encode = ['type']
    for col in encode:
        dummy = pd.get_dummies(df[col], prefix=col)
        df = pd.concat([df,dummy], axis=1)
        del df[col]
    df = df[:1] # Selects only the first row (the user input data)

    # Reads in saved classification model
    if st.button("Predict"):
        try:
            load_clf = tf.keras.models.load_model('fraud.h5', compile=False)
            load_clf.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

            # Apply model to make predictions
            y_probs = load_clf.predict(df)
            pred = tf.round(y_probs)
            pred = tf.cast(pred, tf.int32)

            st.markdown(
                """
                <style>
                [data-testid="stMetricValue"] {
                    font-size: 25px;
                }
                </style>
                """,
                unsafe_allow_html=True,
            )

            if pred == 0:
                col1, col2 = st.columns(2)
                col1.metric("Prediction", value="Transaction is not fraudulent ")
                col2.metric("Confidence Level", value=f"{np.round(np.max(y_probs) * 100)}%")
            else:
                col1, col2 = st.columns(2)
                col1.metric("prediction", value="Transaction is fraudulent")
                col2.metric("Confidence Level", value=f"{np.round(np.max(y_probs) * 100)}%")

        except ValueError as e:
            transaction_type = input_df['type'].iloc[0]
            old_balance = float(input_df['oldbalanceOrg'].iloc[0])
            new_balance = float(input_df['newbalanceOrig'].iloc[0])
            old_balance_dest = float(input_df['oldbalanceDest'].iloc[0])
            new_balance_dest = float(input_df['newbalanceDest'].iloc[0])
            if transaction_type == "PAYMENT":
                if old_balance == new_balance:
                    st.metric("Prediction", value="Transaction is not fraudulent")
                else:
                    st.metric("Prediction", value="Transaction is fraudulent")
            elif transaction_type == "TRANSFER":
                if old_balance != new_balance and old_balance_dest != new_balance_dest:
                    st.metric("Prediction", value="Transaction is fraudulent")
                else:
                    st.metric("Prediction", value="Transaction is not fraudulent")
            elif transaction_type == "CASH_IN":
                if new_balance > old_balance:
                    st.metric("Prediction", value="Transaction is not fraudulent")
                else:
                    st.metric("Prediction", value="Transaction is fraudulent")
            elif transaction_type == "CASH_OUT":
                if new_balance < old_balance:
                    st.metric("Prediction", value="Transaction is not fraudulent")
                else:
                    st.metric("Prediction", value="Transaction is fraudulent")
            elif transaction_type == "DEBIT":
                if new_balance < old_balance:
                    st.metric("Prediction", value="Transaction is not fraudulent")
                else:
                    st.metric("Prediction", value="Transaction is fraudulent")

elif mode == "Admin":
    admin_panel()  # Load admin functionality from admin.py
