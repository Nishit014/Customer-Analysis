import streamlit as st
from PIL import Image
import sklearn
import joblib
model=joblib.load("/Users/mehrotra/python_program/customer_analyzer/term_deposit/model_td")
def Predict(age,contact,duration,education,pdays,poutcome,previous):
#Labels for Education
    if job == "Unemployed" or job == "Student":
        x = 1
    elif job == "Unknown":
        x = 0
    elif job == "Self-employed" or job == "Management" or job == "Technician" or job == "Services":
        x = 2
    else:
        x = 3
    # Labels for contacts
    if contact == "cellular":
        x1 = 0
    else:
        x1 = 1
    # Labels for poutcome
    if poutcome=="Unkown":
        x2 = 0
    elif poutcome == "Failure":
        x2 = 1
    else:
        x2 = 2
    pred = model.predict([[age,x1,duration,x,pdays,x2,previous]])
    return(pred[0])

if __name__ == "__main__":
    # img1 = Image.open("term.jpg")
    # img1 = img1.resize((500, 300))
    # st.image(img1, use_column_width=False)
    html_temp = """
        <div style="background-color:tomato;padding:10px">
        <h2 style="color:white;text-align:center;">Term Deposit Prediction </h2>
        </div>
        """
    st.markdown(html_temp, unsafe_allow_html=True)
    age = st.number_input("Age",min_value=18)
    job = st.selectbox("Job Type ",('Unemployed', 'Student', 'Services', 'Technician','Management', 'Self-employed', 'Blue-collar', 'Entrepreneur','Retired','Unknown'))
    contact= st.radio("Contact",("cellular","telephone"))
    duration = st.number_input("Enter call durations in seconds",step=1)
    previous=st.number_input("Number of contacts performed before this campaign for this client",step=1)
    pdays = st.number_input("Number of days that passed by after the client was last contacted in previous campaign",step=1)
    poutcome=st.radio("Outcome of the previous marketing campaign",('Failure','Success','Unknown'))
    if st.button("Predict"):
        result = Predict(age,contact,duration,job,pdays,poutcome,previous)
        if result == 1:
            st.success("Their is a high possibility that this person will go for term deposit")
        else:
            st.success("Their is a high possibility that this person will not go for term deposit")


