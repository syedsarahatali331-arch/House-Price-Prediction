import streamlit as st, pandas as pd, joblib
st.title("House Price Predictor")
model=joblib.load("model.pkl")
area=st.number_input("Area (sq ft)",500,10000,1500)
bed=st.number_input("Bedrooms",1,10,3); bath=st.number_input("Bathrooms",1,10,2)
age=st.number_input("House Age",0,80,10); loc=st.selectbox("Location",["Suburban","Urban","Prime"])
if st.button("Predict Price"):
    x=pd.DataFrame([{"Area_sqft":area,"Bedrooms":bed,"Bathrooms":bath,"House_Age":age,"Location":loc}])
    st.metric("Predicted price",f"₹{model.predict(x)[0]:,.0f}")
