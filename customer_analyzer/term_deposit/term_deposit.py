import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm
import scipy.stats as stats
from sklearn.model_selection import train_test_split
from sklearn import metrics
from statsmodels.stats.outliers_influence import variance_inflation_factor
from patsy import dmatrices

df=pd.read_csv("/Users/mehrotra/python_program/customer_analyzer/term_deposit/new_train.csv")
df

df["previous"].unique()
df.info()

df["y"]=df["y"].map({"yes":1,"no":0})

sns.scatterplot("duration","y",hue="marital",data=df)
sns.scatterplot("y","duration",hue="marital",data=df)
sns.barplot("y","duration",hue="marital",data=df)

df.groupby("job")["y"].sum().plot(kind="bar")
df.groupby("education")["y"].sum().plot(kind="bar")
df.groupby("month")["y"].sum().plot(kind="bar")
df.groupby("marital")["y"].sum().plot(kind="bar")
sns.countplot("job",hue="y",data=df)

df["y"].value_counts()
df.drop(["month","day_of_week"],axis=1,inplace=True)
df["job"].unique()
df["job"]=df["job"].map({"unemployed":1,"unknown":0,"housemaid":1,"student":1,"services":2,"technician":2,"self-employed":2,"management":2,"admin.":3,"retired":3,"entrepreneur":3,"blue-collar":3})
df["marital"].unique()
df["marital"]=df["marital"].map({"married":2,"divorced":1,"single":1,"unknown":1})
df["education"].unique()
df["education"]=df["education"].map({"illiterate":0,"basic.4y":0,"basic.6y":0,"basic.9y":0,"high.school":1,"professional.course":2,"university.degree":3,"unknown":1})
df["education"].value_counts()
df["default"]=df["default"].map({"unknown":0,"no":0,"yes":1})
df["housing"]=df["housing"].map({"unknown":0,"no":0,"yes":1})
df["loan"]=df["loan"].map({"unknown":0,"no":0,"yes":1})
df["contact"]=df["contact"].map({"cellular":0,"telephone":1})
df["pdays"]=df["pdays"].replace(999,0)
df["poutcome"]=df["poutcome"].map({"nonexistent":0,"failure":1,"success":2})
df

from sklearn.feature_selection import mutual_info_classif
# determine the mutual information
mutual_info = mutual_info_classif(df[df.columns.difference(["y"])], df["y"])
mutual_info = pd.Series(mutual_info)
mutual_info.index = df[df.columns.difference(["y"])].columns
mutual_info.sort_values(ascending=False)
mutual_info.sort_values(ascending=False).plot.bar(figsize=(20, 8))

from sklearn.feature_selection import SelectKBest
#No we Will select the  top 5 important features
sel_five_cols = SelectKBest(mutual_info_classif, k=7)
sel_five_cols.fit(df[df.columns.difference(["y"])], df["y"])
columns=df[df.columns.difference(["y"])].columns[sel_five_cols.get_support()]
df1=df[columns]

y=df["y"]

df1

X_train, X_test, y_train, y_test = train_test_split(df1, y, test_size=0.3, random_state=5)

from imblearn.under_sampling import RandomUnderSampler 
from collections import Counter

rus = RandomUnderSampler(random_state=500)
X_under_train, y_under_train = rus.fit_resample(X_train, y_train)

print('Original dataset shape {}'.format(Counter(y_train)))
print('Undersampled dataset shape {}'.format(Counter(y_under_train)))

from sklearn.ensemble import RandomForestClassifier
pargrid_rf = {'n_estimators': [50, 60, 70, 80, 90, 100],
                  'max_features': [5,6,7,8,9,10,11,12]}

from sklearn.model_selection import GridSearchCV
gscv_rf = GridSearchCV(estimator=RandomForestClassifier(), 
                        param_grid=pargrid_rf, 
                        cv=10,
                        verbose=True, n_jobs=-1)

gscv_results = gscv_rf.fit(X_under_train,y_under_train)
print(gscv_results.best_params_)

radm_clf = RandomForestClassifier(n_estimators=80, max_features=7, n_jobs=-1)
radm_clf.fit( X_under_train,y_under_train )

tree_cm = metrics.confusion_matrix( y_under_train ,radm_clf.predict(X_under_train) )
tree_cm

metrics.roc_auc_score(y_train,radm_clf.predict(X_train))
print(metrics.accuracy_score(y_train,radm_clf.predict(X_train)))
print(metrics.accuracy_score(y_test,radm_clf.predict(X_test)))

x=radm_clf.predict([[49,0,560,2,0,0,0]])
x[0]

import joblib
joblib.dump(radm_clf,"model_td")

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
print(cross_val_score(LogisticRegression(),X_train, y_train))
print(cross_val_score(DecisionTreeClassifier(),X_train, y_train))
print(cross_val_score(SVC(),X_train, y_train))
print(cross_val_score(RandomForestClassifier(),X_train, y_train))











#This database contains the details of the customers who have subscribed to the service.
# 1 - Age (numeric)
# 2 - Job : type of job (categorical)
# 3 - Marital : marital status (categorical)
# 4 - Education (categorical)
# 5 - Default: has credit in default? (categorical)
# 6 - Housing: has housing loan? (categorical)
# 7 - Loan: has personal loan? (categorical)
# 8 - Contact: contact communication type (categorical)
# 9 - Month: last contact month of year (categorical)
# 10 - Day_of_week: last contact day of the week (categorical)
# 11 - Duration: last contact duration, in seconds (numeric)
# 12 - Campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)
# 13 - Pdays: number of days that passed by after the client was last contacted from a previous campaign (numeric; 999 means client was not previously contacted)
# 14 - Previous: number of contacts performed before this campaign and for this client (numeric)
# 15 - Poutcome: outcome of the previous marketing campaign (categorical)
# 16 - y: labels (binary)
