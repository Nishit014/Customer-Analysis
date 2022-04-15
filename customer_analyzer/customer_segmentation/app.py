import streamlit as st
import sklearn
import joblib
from PIL import Image
model=joblib.load("/Users/mehrotra/python_program/customer_analyzer/customer_segmentation/model_cs.joblib")
def Predict(BALANCE, BALANCE_FREQUENCY, PURCHASES, ONEOFF_PURCHASES,
       INSTALLMENTS_PURCHASES, CASH_ADVANCE, PURCHASES_FREQUENCY,
       ONEOFF_PURCHASES_FREQUENCY, PURCHASES_INSTALLMENTS_FREQUENCY,
       CASH_ADVANCE_FREQUENCY, CASH_ADVANCE_TRX, PURCHASES_TRX,
       CREDIT_LIMIT, PAYMENTS, MINIMUM_PAYMENTS, PRC_FULL_PAYMENT,
       TENURE):
    pred = model.predict([[BALANCE, BALANCE_FREQUENCY, PURCHASES, ONEOFF_PURCHASES,
       INSTALLMENTS_PURCHASES, CASH_ADVANCE, PURCHASES_FREQUENCY,
       ONEOFF_PURCHASES_FREQUENCY, PURCHASES_INSTALLMENTS_FREQUENCY,
       CASH_ADVANCE_FREQUENCY, CASH_ADVANCE_TRX, PURCHASES_TRX,
       CREDIT_LIMIT, PAYMENTS, MINIMUM_PAYMENTS, PRC_FULL_PAYMENT,
       TENURE]])
    return(pred[0])

if __name__ == "__main__":
    # img1 = Image.open('img.png')
    # img1 = img1.resize((500, 300))
    # st.image(img1, use_column_width=False)
    html_temp = """
            <div style="background-color:tomato;padding:10px">
            <h2 style="color:white;text-align:center;"> Customer Segmentation </h2>
            </div>
            """
    st.markdown(html_temp, unsafe_allow_html=True)
    BALANCE = st.number_input("Balance amount left in their account to make purchases")
    BALANCE_FREQUENCY = st.number_input("How frequently the Balance is updated(Frequency between 0 and 1)")
    PURCHASES = st.number_input("PURCHASES")
    ONEOFF_PURCHASES = st.number_input("ONEOFFPURCHASES")
    INSTALLMENTS_PURCHASES = st.number_input("INSTALLMENTS_PURCHASES")
    CASH_ADVANCE = st.number_input("Cash in advance given by the user")
    PURCHASES_FREQUENCY = st.number_input("How frequently the Purchases are being made, score between(Frequency between 0 and 1)")
    ONEOFF_PURCHASES_FREQUENCY = st.number_input("How frequently Purchases are happening in one-go(Frequency between 0 and 1")
    PURCHASES_INSTALLMENTS_FREQUENCY = st.number_input("How frequently purchases in installments are being done(Frequency between 0 and 1)")
    CASH_ADVANCE_FREQUENCY = st.number_input("How frequently the cash in advance being paid(Frequency between 0 and 1)")
    CASH_ADVANCE_TRX = st.number_input("Number of Transactions made with cash in advance")
    PURCHASES_TRX = st.number_input("Number of purchase transactions made")
    CREDIT_LIMIT = st.number_input("CREDITLIMIT")
    PAYMENTS = st.number_input("PAYMENTS")
    MINIMUM_PAYMENTS = st.number_input("MINIMUM_PAYMENTS")
    PRC_FULL_PAYMENT = st.number_input("Percent of full payment paid by user")
    TENURE = st.number_input("Tenure of credit card service for user",step=1)
    if st.button("Predict"):

        result=Predict(BALANCE, BALANCE_FREQUENCY, PURCHASES, ONEOFF_PURCHASES,
        INSTALLMENTS_PURCHASES, CASH_ADVANCE, PURCHASES_FREQUENCY,
        ONEOFF_PURCHASES_FREQUENCY, PURCHASES_INSTALLMENTS_FREQUENCY,
        CASH_ADVANCE_FREQUENCY, CASH_ADVANCE_TRX, PURCHASES_TRX,
        CREDIT_LIMIT, PAYMENTS, MINIMUM_PAYMENTS, PRC_FULL_PAYMENT,
        TENURE)
        if result==0:
            st.write("Findings")
            st.write("People from this group have balance in their account but does not use credit card or debit cards for transactions.")
            st.write("Recommendation")
            st.write("Banks can give cashback offers on cards to persuade the customers")
        elif result==1:
            st.write("Findings")
            st.write("People from this group  neither have much balance in their account nor they use cards for transaction")
            st.write("Recommendation")
            st.write("Banks can approach this group and let them know the advantages of using accounts payments ")
        elif result==2:
            st.write("Findings")
            st.write("People from this group  have less balance in their accounts and they use installements schemes  for transactions")
            st.write("Recommendation")
            st.write("Banks can approach this group let them know about term deposits and deposit schemes ")
        else:
            st.write("Findings")
            st.write("People from this group have very high balance in their accounts and the use of cards is very frequent this group is most conducive for the bank")
            st.write("Recommendation")
            st.write("Banks should give them special offers so that they keep using their accounts for payments")



