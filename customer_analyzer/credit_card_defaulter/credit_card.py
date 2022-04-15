import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report,confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

df = pd.read_csv('credit-card-default.csv')
df.head()
df.info()
df.columns
df.SEX.value_counts()
df.EDUCATION.value_counts()
df.BILL_AMT1.value_counts()

df['EDUCATION'].replace([0, 6], 5, inplace=True)
df.EDUCATION.value_counts()

df.MARRIAGE.value_counts()
df['MARRIAGE'].replace(0, 3, inplace=True)
df.MARRIAGE.value_counts()

df.PAY_2.value_counts()
df.PAY_0.value_counts()



df.drop('ID',axis=1, inplace=True)


X = df.drop('defaulted',axis=1)

y = df['defaulted']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)


rfc = RandomForestClassifier()

rfc.fit(X_train,y_train)

predictions = rfc.predict(X_test)



print(classification_report(y_test,predictions))


print(confusion_matrix(y_test,predictions))

print(accuracy_score(y_test,predictions))

n_folds = 5

parameters = {'max_depth': range(2, 20, 5)}

rf = RandomForestClassifier()


rf = GridSearchCV(rf, parameters, 
                    cv=n_folds, 
                   scoring="accuracy",
                 return_train_score=True)
rf.fit(X_train, y_train)



scores = rf.cv_results_
pd.DataFrame(scores).head()

plt.figure()
plt.plot(scores["param_max_depth"], # x-axis
         scores["mean_train_score"], # y-axis
         label="training accuracy")
plt.plot(scores["param_max_depth"], 
         scores["mean_test_score"], 
         label="test accuracy")
plt.xlabel("max_depth")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

n_folds = 5

parameters = {'n_estimators': range(100, 1500, 400)}

rf = RandomForestClassifier(max_depth=4)


rf = GridSearchCV(rf, parameters, 
                    cv=n_folds, 
                   scoring="accuracy",
                 return_train_score=True)
rf.fit(X_train, y_train)



scores = rf.cv_results_
pd.DataFrame(scores).head()


plt.figure()
plt.plot(scores["param_n_estimators"], 
         scores["mean_train_score"], 
         label="training accuracy")
plt.plot(scores["param_n_estimators"], 
         scores["mean_test_score"], 
         label="test accuracy")
plt.xlabel("n_estimators")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

n_folds = 5

parameters = {'max_features': [4, 8, 14, 20, 24]}

rf = RandomForestClassifier(max_depth=4)

rf = GridSearchCV(rf, parameters, 
                    cv=n_folds, 
                   scoring="accuracy",
                 return_train_score=True)
rf.fit(X_train, y_train)



scores = rf.cv_results_
pd.DataFrame(scores).head()


plt.figure()
plt.plot(scores["param_max_features"], 
         scores["mean_train_score"], 
         label="training accuracy")
plt.plot(scores["param_max_features"], 
         scores["mean_test_score"], 
         label="test accuracy")
plt.xlabel("max_features")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

n_folds = 5


parameters = {'min_samples_leaf': range(100, 400, 50)}


rf = RandomForestClassifier()

rf = GridSearchCV(rf, parameters, 
                    cv=n_folds, 
                   scoring="accuracy",
                 return_train_score=True)
rf.fit(X_train, y_train)


scores = rf.cv_results_
pd.DataFrame(scores).head()


plt.figure()
plt.plot(scores["param_min_samples_leaf"], 
         scores["mean_train_score"], 
         label="training accuracy")
plt.plot(scores["param_min_samples_leaf"], 
         scores["mean_test_score"], 
         label="test accuracy")
plt.xlabel("min_samples_leaf")
plt.ylabel("Accuracy")
plt.legend()
plt.show()



n_folds = 5

parameters = {'min_samples_split': range(200, 500, 50)}

rf = RandomForestClassifier()

rf = GridSearchCV(rf, parameters, 
                    cv=n_folds, 
                   scoring="accuracy",
                 return_train_score=True)
rf.fit(X_train, y_train)

scores = rf.cv_results_
pd.DataFrame(scores).head()

plt.figure()
plt.plot(scores["param_min_samples_split"], 
         scores["mean_train_score"], 
         label="training accuracy")
plt.plot(scores["param_min_samples_split"], 
         scores["mean_test_score"], 
         label="test accuracy")
plt.xlabel("min_samples_split")
plt.ylabel("Accuracy")
plt.legend()
plt.show()


param_grid = {
    'max_depth': [4,8,10],
    'min_samples_leaf': range(100, 400, 50),
    'min_samples_split': range(200, 500, 50),
    'n_estimators': [100,200, 300], 
    'max_features': [5, 10]
}

rf = RandomForestClassifier()

grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
                          cv = 3, n_jobs = -1,verbose = 1)

grid_search.fit(X_train, y_train)

print('We can get accuracy of',grid_search.best_score_,'using',grid_search.best_params_)

type(grid_search.best_params_)
print(grid_search.best_params_)

rfc = RandomForestClassifier(bootstrap=True,
                             max_depth=4,
                             min_samples_leaf=100, 
                             min_samples_split=200,
                             max_features=10,
                             n_estimators=300)

rfc.fit(X_train,y_train)

predictions = rfc.predict(X_test)
print(predictions)
# Let's check the report of our default model
print(classification_report(y_test,predictions))

# Printing confusion matrix
print(confusion_matrix(y_test,predictions))

print(accuracy_score(y_test,predictions))
rfc.predict([[20000,2,2,1,24,2,2,-1,-1,-2,-2,3913,3102,689,0,0,0,0,689,0,0,0,0]])

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
print(cross_val_score(LogisticRegression(),X_train,y_train))
print(cross_val_score(DecisionTreeClassifier(),X_train,y_train))
print(cross_val_score(SVC(),X_train,y_train))
print(cross_val_score(RandomForestClassifier(),X_train,y_train))

import joblib
joblib.dump(rfc,"model_ccd")









# ##### Dataset Information
# This dataset contains information on default payments, demographic factors, credit data, history of payment, and bill statements of credit card clients in Taiwan from April 2005 to September 2005.

# ##### There are 25 variables:
# 
# - **ID**: ID of each client
# - **LIMIT_BAL**: Amount of given credit in NT dollars (includes individual and family/supplementary credit
# - **SEX**: Gender (1=male, 2=female)
# - **EDUCATION**: (1=graduate school, 2=university, 3=high school, 4=others, 5=unknown, 6=unknown)
# - **MARRIAGE**: Marital status (1=married, 2=single, 3=others)
# - **AGE**: Age in years
# - **PAY_0**: Repayment status in September, 2005 (-1=pay duly, 1=payment delay for one month, 2=payment delay for two months, … 8=payment delay for eight months, 9=payment delay for nine months and above)
# - **PAY_2**: Repayment status in August, 2005 (scale same as above)
# - **PAY_3**: Repayment status in July, 2005 (scale same as above)
# - **PAY_4**: Repayment status in June, 2005 (scale same as above)
# - **PAY_5**: Repayment status in May, 2005 (scale same as above)
# - **PAY_6**: Repayment status in April, 2005 (scale same as above)
# - **BILL_AMT1**: Amount of bill statement in September, 2005 (NT dollar)
# - **BILL_AMT2**: Amount of bill statement in August, 2005 (NT dollar)
# - **BILL_AMT3**: Amount of bill statement in July, 2005 (NT dollar)
# - **BILL_AMT4**: Amount of bill statement in June, 2005 (NT dollar)
# - **BILL_AMT5**: Amount of bill statement in May, 2005 (NT dollar)
# - **BILL_AMT6**: Amount of bill statement in April, 2005 (NT dollar)
# - **PAY_AMT1**: Amount of previous payment in September, 2005 (NT dollar)
# - **PAY_AMT2**: Amount of previous payment in August, 2005 (NT dollar)
# - **PAY_AMT3**: Amount of previous payment in July, 2005 (NT dollar)
# - **PAY_AMT4**: Amount of previous payment in June, 2005 (NT dollar)
# - **PAY_AMT5**: Amount of previous payment in May, 2005 (NT dollar)
# - **PAY_AMT6**: Amount of previous payment in April, 2005 (NT dollar)
# - **default.payment.next.month**: Default payment (1=yes, 0=no)
