import streamlit as st
import sklearn
import joblib

model=joblib.load("/Users/mehrotra/python_program/customer_analyzer/credit_card_defaulter/model_ccd")

def Predict(LIMIT_BAL,SEX,EDUCATION,MARRIAGE,AGE,PAY_0,PAY_2,PAY_3,PAY_4,PAY_5,PAY_6,
        BILL_AMT1,BILL_AMT2,BILL_AMT3,BILL_AMT4,BILL_AMT5,BILL_AMT6,PAY_AMT1,
        PAY_AMT2,PAY_AMT3,PAY_AMT4,PAY_AMT5,PAY_AMT6):
    pred=model.predict([[LIMIT_BAL,SEX,EDUCATION,MARRIAGE,AGE,PAY_0,PAY_2,PAY_3,PAY_4,PAY_5,PAY_6,
        BILL_AMT1,BILL_AMT2,BILL_AMT3,BILL_AMT4,BILL_AMT5,BILL_AMT6,PAY_AMT1,
        PAY_AMT2,PAY_AMT3,PAY_AMT4,PAY_AMT5,PAY_AMT6]])

    return(pred[0])

html_temp = """
            <div style="background-color:tomato;padding:10px">
            <h2 style="color:white;text-align:center;"> Credit Card Defaulter </h2>
            </div>
            """
st.markdown(html_temp, unsafe_allow_html=True)
st.markdown(' ')
st.markdown('Demographic Information')
lb=st.number_input("Limited balance:",step=100)
s=st.selectbox("Sex:",('Male','Female'))
if s=='Male':
    s=1
else:
    s=2
ed=st.selectbox('Education:',('Under Graduate','Graduate','Post Graduate'))
if ed=='Post Graduate':
    ed=1
elif ed=='Graduate':
    ed=2
else:
    ed=3

age,ms=st.columns(2)
age=age.number_input('Enter the age:',min_value=18)
ms=ms.selectbox('Marital Status:',('Single','Married','Divorced'))
if ms=='Married':
    ms=1
elif ms=='Single':
    ms=2
else:
    ms=3

st.markdown(' ')
st.markdown('Behaveur Information')
st.markdown('Previous Months Repayment Status (-1=paid duly, 1= 1 month delay, ...., 6= 6 months delay):')
# st.text('(-1=paid duly,1= 1 month delay,....6= 6 months delay)')

m01,m02,m03,m04,m05,m06=st.columns(6)
m1=m01.text_input('Last Month:')
m2=m02.text_input('2nd Last Month:')
m3=m03.text_input('3rd Last Month:')
m4=m04.text_input('4th Last Month:')
m5=m05.text_input('5th Last Month:')
m6=m06.text_input('6th Last Month:')

st.markdown('Bill Amounts')
b01,b02,b03,b04,b05,b06=st.columns(6)
b1=b01.text_input('Last Month:  ')
b2=b02.text_input('2nd Last Month:  ')
b3=b03.text_input('3rd Last Month:  ')
b4=b04.text_input('4th Last Month:  ')
b5=b05.text_input('5th Last Month:  ')
b6=b06.text_input('6th Last Month:  ')

st.markdown('Previous Payment Status')
p01,p02,p03,p04,p05,p06=st.columns(6)
p1=p01.text_input('Last Month: ')
p2=p02.text_input('2nd Last Month: ')
p3=p03.text_input('3rd Last Month: ')
p4=p04.text_input('4th Last Month: ')
p5=p05.text_input('5th Last Month: ')
p6=p06.text_input('6th Last Month: ')

if st.button("Predict"):
    # st.write(lb,s,ed,ms,age,m1,m2,m3,m4,m5,m6,b1,b2,b3,b4,b5,b6,p1,p2,p3,p4,p5,p6)
    result=Predict(lb,s,ed,ms,age,m1,m2,m3,m4,m5,m6,b1,b2,b3,b4,b5,b6,p1,p2,p3,p4,p5,p6)
    if result==1:
        st.success('The customer will default in the next month')
    else:
        st.success('The customer will not default in the next month')