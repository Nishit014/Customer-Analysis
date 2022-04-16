import streamlit as st
from PIL import Image
import joblib
import sklearn


model = joblib.load("/Users/mehrotra/python_program/customer_analyzer/loan_prediction/model_lp")
def Prediction(age, ed, employ, debtinc, creddebt,othdebt):
    pred = model.predict_proba([[age, ed, employ, debtinc, creddebt, othdebt]])
    pred_default=pred[0][1]
    print(pred[0][1])
    return(pred_default)

if __name__ == "__main__":
    # img1 = Image.open('/Users/mehrotra/python_program/Banking-app-main/Bank_Loan/bans.jpg')
    # img1 = img1.resize((500, 300))
    # st.image(img1, use_column_width=False)
    html_temp = """
        <div style="background-color:tomato;padding:10px">
        <h2 style="color:white;text-align:center;">Bank Loan Prediction</h2>
        </div>
        """
    st.markdown(html_temp, unsafe_allow_html=True)
    age = st.number_input("Age", min_value=18, step=1)
    # ed = st.number_input(" Years of education", min_value=0, max_value=5, step=1)
    ed=st.selectbox('Education:',('Under Graduate','Graduate','Post Graduate'))
    if ed=='Under Graduate':
        ed=3
    elif ed=='Graduate':
        ed=2
    else:
        ed=1
    employ = st.number_input("Employed for how many years",step=0.1)
    # address= st.number_input("Years at current address",step=0.1)
    # income = st.number_input("Income",step=1)
    debtinc = st.number_input("Debt to income ratio")
    creddebt = st.number_input("Credit debt")
    othdebt = st.number_input("Other debt")
    if st.button("Predict"):
        # print(income/100)
        # result=Prediction(age, ed, employ, address, income/10000, debtinc, creddebt, othdebt)
        result=Prediction(age, ed, employ, debtinc, creddebt, othdebt)
        if result> 0.2476080913770519:
            st.success("The person will Default its our recommendation not to give this person loan")
        else:
            st.success("The person will not Default its our recommendation to give this person loan")

