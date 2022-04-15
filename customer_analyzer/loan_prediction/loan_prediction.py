import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm
import scipy.stats as stats
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from statsmodels.stats.outliers_influence import variance_inflation_factor
# from patsy import dmatrices

df1=pd.read_csv('/Users/mehrotra/python_program/customer_analyzer/loan_prediction/bankloans.csv')
df1.columns
df1.info()
df1.head()

bankloans_existing = df1[df1.default.isnull()==0]
bankloans_new = df1[df1.default.isnull()==1]

def var_summary(x):
    return pd.Series([x.count(), x.isnull().sum(), x.sum(), x.mean(), x.median(),  x.std(), x.var(), x.min(), x.dropna().quantile(0.01), x.dropna().quantile(0.05),x.dropna().quantile(0.10),x.dropna().quantile(0.25),x.dropna().quantile(0.50),x.dropna().quantile(0.75), x.dropna().quantile(0.90),x.dropna().quantile(0.95), x.dropna().quantile(0.99),x.max()],index=['N', 'NMISS', 'SUM', 'MEAN','MEDIAN', 'STD', 'VAR', 'MIN', 'P1' , 'P5' ,'P10' ,'P25' ,'P50' ,'P75' ,'P90' ,'P95' ,'P99' ,'MAX'])

num_summary=bankloans_existing.apply(var_summary).T
num_summary

bankloans_numeric=bankloans_existing.drop("default",axis=1)
y=bankloans_existing["default"]

sns.boxplot(data=bankloans_numeric)

def IQR_capping(x):
    IQR=x.quantile(0.75)-x.quantile(0.25)
    upper_bridge=x.quantile(0.75)+1.5*(IQR)
    lower_bridge=x.quantile(0.25)-1.5*(IQR)
    x = x.clip(upper=upper_bridge)
    x = x.clip(lower=lower_bridge)
    return x

bankloans_numeric=bankloans_numeric.apply(IQR_capping)  

bankloans_numeric.describe()

#After outlier removal
sns.boxplot(data=bankloans_numeric)
sns.heatmap(bankloans_numeric.corr())
sns.scatterplot(x="age",y="income",data=bankloans_numeric)
sns.pairplot(data=bankloans_existing)


# from
# from statsmodels.stats.outliers_influence import variance_inflation_factor
# variables = bankloans_numeric

# # we create a new data frame which will include all the VIFs
# # note that each variable has its own variance inflation factor as this measure is variable specific (not model specific)
# # we do not include categorical values for mulitcollinearity as they do not provide much information as numerical ones do
# vif = pd.DataFrame()

# # here we make use of the variance_inflation_factor, which will basically output the respective VIFs 
# vif["VIF"] = [variance_inflation_factor(variables.values, i) for i in range(variables.shape[1])]
# #Finally, I like to include names so it is easier to explore the result
# vif["Features"] = bankloans_numeric.columns
# vif

# # bankloans_numeric.drop(["income","address"],axis=1,inplace=True)
# till
bankloans_numeric

train_x,test_x,train_Y,test_Y = train_test_split(bankloans_numeric,y,test_size=0.3, random_state=42)

logreg = LogisticRegression()
logreg.fit( train_x, train_Y)

#Finding optimal cutoff
bankloans_pred_test = pd.DataFrame( { 'actual':  test_Y,'predicted': logreg.predict( test_x) } )
bankloans_pred_train = pd.DataFrame( { 'actual':  train_Y,'predicted': logreg.predict( train_x ) } )
bankloans_pred_train

cm = metrics.confusion_matrix( bankloans_pred_train.actual,bankloans_pred_train.predicted)#, [1,0] )
cm

sns.heatmap(cm, annot=True,  fmt='.2f', xticklabels = ["default", "Not default"] , yticklabels = ["default", "Not defsult"] )
plt.ylabel('True label')
plt.xlabel('Predicted label')

bankloans_pred_train.reset_index(inplace=True)
bankloans_pred_train

predict_proba = pd.DataFrame( logreg.predict_proba( train_x ) )
predict_proba.drop(0,axis=1,inplace=True)
predict_proba

bankloans_pred_train
bankloans_pred = pd.concat([bankloans_pred_train, predict_proba],axis = 1 )
bankloans_pred.columns = [ "index",'actual', 'predicted', 'probability_default']
bankloans_pred

auc_score = metrics.roc_auc_score( bankloans_pred.actual, bankloans_pred.probability_default  )
round( float( auc_score ), 2 )

fpr, tpr, thresholds = metrics.roc_curve( bankloans_pred.actual,bankloans_pred.probability_default,drop_intermediate = False )

plt.figure(figsize=(6, 4))
plt.plot( fpr, tpr, label='ROC curve (area = %0.2f)' % auc_score )
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate or [1 - True Negative Rate]')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()

print(thresholds[0:10])
print(fpr[0:10])
print(tpr[0:10])

print(tpr[np.abs(tpr - 0.8).argmin()])
cutoff_probability= thresholds[(np.abs(tpr - 0.8)).argmin()]
cutoff_probability

bankloans_pred['new_labels'] = bankloans_pred["probability_default"].map( lambda x: 1 if x >=cutoff_probability else 0 )
bankloans_pred

predict_proba_test = pd.DataFrame( logreg.predict_proba( test_x ) )
predict_proba_test.drop(0,axis=1,inplace=True)

bankloans_pred_test.reset_index(inplace=True)
bankloans_pred_test=pd.concat([bankloans_pred_test,predict_proba_test],axis=1)
metrics.roc_auc_score(bankloans_pred_test["actual"],bankloans_pred_test[1])

import joblib
joblib.dump(logreg,"model_lp1")

logreg.predict_proba([[25,3,4,3,68,5,7,6]])[0][1]

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
print(cross_val_score(LogisticRegression(),train_x, train_Y))
print(cross_val_score(DecisionTreeClassifier(),train_x, train_Y))
print(cross_val_score(SVC(),train_x, train_Y))
print(cross_val_score(RandomForestClassifier(),train_x, train_Y))








# age - Age of Customer
# ed - Eductation level of customer
# employ: Tenure with current employer (in years)
# address: Number of years in same address
# income: Customer Income
# debtinc: Debt to income ratio
# creddebt: Credit to Debt ratio
# othdebt: Other debts
# default: Customer defaulted in the past (1= defaulted, 0=Never defaulted)














# References
# [1] Aslam U, Aziz H I T, Sohail A and Batcha N K 2019 An empirical study on loan default
# prediction models Journal of Computational and Theoretical Nanoscience 16 pp 3483–8
# [2] Li Y 2019 Credit risk prediction based on machine learning methods The 14th Int. Conf. on
# Computer Science & Education (ICCSE) pp 1011–3
# [3] Ahmed M S I and Rajaleximi P R 2019 An empirical study on credit scoring and credit scorecard
# for financial institutions Int. Journal of Advanced Research in Computer Engineering & Technol.
# (IJARCET) 8 275–9
# [4] Zhu L, Qiu D, Ergu D, Ying C and Liu K 2019 A study on predicting loan default based on the
# random forest algorithm The 7th Int. Conf. on Information Technol. and Quantitative
# Management (ITQM) 162 pp 503–13
# [5] Ghatasheh N 2014 Business analytics using random forest trees for credit risk prediction: a
# comparison study Int. Journal of Advanced Science and Technol. 72 pp 19–30
# [6] Breeden J L 2020 A survey of machine learning in credit risk
# [7] Madane N and Nanda S 2019 Loan prediction analysis using decision tree Journal of The Gujarat
# Research Society 21 p p 214–21
# [8] Supriya P, Pavani M, Saisushma N, Kumari N V and Vikas K 2019 Loan prediction by using
# machine learning models Int. Journal of Engineering and Techniques 5 pp144–8
# [9] Amin R K, Indwiarti and Sibaroni Y 2015 Implementation of decision tree using C4.5 algorithm
# in decision making of loan application by debtor (case study: bank pasar of yogyakarta special
# region) The 3rd Int. Conf. on Information and Communication Technol. (ICoICT) pp 75–80
# [10] Jency X F, Sumathi V P and Sri J S 2018 An exploratory data analysis for loan prediction based
# on nature of the clients Int. Journal of Recent Technol. and Engineering (IJRTE) 7 pp 176–9 
# [11] Shoumo S Z H, Dhruba M I M, Hossain S, Ghani N H, Arif H and Islam S 2019 Application of
# machine learning in credit risk assessment: a prelude to smart banking TENCON 2019 – 2019
# IEEE Region 10 Conf. (TENCON) pp 2023–8
# [12] Addo P M, Guegan D and Hassani B 2018 Credit risk analysis using machine and deep learning
# models Risks 6 p 38
# [13] Hamid A J and Ahmed T M 2016 Developing prediction model of loan risk in banks using data
# mining Machine Learning and Applications: An Int. Journal (MLAIJ) 3 pp 1–9
# [14] Kacheria A, Shivakumar N, Sawkar S and Gupta A 2016 Loan sanctioning prediction system Int.
# Journal of Soft Computing and Engineering (IJSCE) 6 pp 50–3
# [15] Vojtek M and Kocenda E 2006 Credit scoring methods Finance a uver - Czech Journal of
# Economics and Finance 56 pp 152–167
# [16] Russel S and Norvig P 1995 Artificial intelligence - a modern approach
# [17] Alshouiliy K, Alghamdi A and Agrawal D P 2020 AzureML based analysis and prediction loan
# borrowers creditworthy The 3rd Int. Conf. on Information and Computer Technologies (ICICT) 1
# pp 302–6
# [18] Li M, Mickel A and Taylor S 2018, “Should this loan be approved or denied?”: a large dataset
# with class assignment guidelines Journal of Statistics Education 26 pp 55–66
# [19] Vaidya A 2017 Predictive and probabilistic approach using logistic regression: application to
# prediction of loan approval The 8th Int. Conf. on Computing, Communication and Networking
# Technologies (ICCCNT) 1 pp 1–6
# [20] Murphy K P 2012 Machine learning: a probabilistic approach 