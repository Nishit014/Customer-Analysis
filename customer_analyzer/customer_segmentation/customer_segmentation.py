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
from patsy import dmatrices
from sklearn.cluster import KMeans
# center and scale the data
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

df=pd.read_csv("/Users/mehrotra/python_program/customer_analyzer/customer_segmentation/Segmentation.csv")
df
df.info()
df.drop("CUST_ID",axis=1,inplace=True)
df.describe()

sns.boxplot(data=df)

def outlier_capping(x):
    x = x.clip(upper=x.quantile(0.95))
    x = x.clip(lower=x.quantile(0.01))
    return x

df=df.apply(outlier_capping)

df["CREDIT_LIMIT"]=df["CREDIT_LIMIT"].fillna(df["CREDIT_LIMIT"].mean())
df["MINIMUM_PAYMENTS"]=df["MINIMUM_PAYMENTS"].fillna(df["MINIMUM_PAYMENTS"].mean())
df.info()

plt.rcParams['figure.figsize'] = 15, 12
plt.rcParams['axes.grid'] = True
sns.heatmap(df.corr(),annot=False)

#Applying Standard scaler
std_scale = StandardScaler()
df_scaled = std_scale.fit_transform(df)
df_scaled

#Applying PCA
pc = PCA(n_components=17)
pc.fit(df_scaled)
pc.explained_variance_
var= pc.explained_variance_ratio_
var
var1=np.cumsum(np.round(pc.explained_variance_ratio_, decimals=4)*100)
var1

df1=pd.DataFrame({"eigen values":pc.explained_variance_,"explained variance":var1})
df1

pc_final=PCA(n_components=4).fit(df_scaled)
pc_final.explained_variance_
reduced_var=pc_final.fit_transform(df_scaled)
reduced_var

reduced=pd.DataFrame(reduced_var,columns=["C1","C2","C3","C4"])
reduced

wcss=[]
for i in range (1,11):
    kmeans=KMeans(n_clusters=i,init='k-means++',random_state=42)
    kmeans.fit(reduced)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

kmeans = KMeans(n_clusters = 4, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(reduced)
len(y_kmeans)

df["clusters"]=kmeans.labels_
df

metrics.silhouette_score(df_scaled, kmeans.labels_)

sns.scatterplot(x = 'BALANCE', y = 'PURCHASES', data = df, hue = 'clusters')
sns.scatterplot(x = 'PURCHASES', y = 'BALANCE', data = df, hue = 'clusters')

df.clusters.value_counts()

from joblib import dump, load
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

model = Pipeline(steps=[
    ('scaler', std_scale),
    ('pca', pc_final),
    ('ridge', kmeans)
])

dump(model, 'model_cs.joblib')
model = load('model_cs.joblib')

model.predict([[3202.467416,0.909091,0.00,0.00,0.00,4647.169122,0.000000,0.000000,0.000000,0.250000,4,0,7000.0,4103.032597,1072.340217,0.222222,12]])[0]

cluster1=df[df["clusters"]==0]
cluster2=df[df["clusters"]==1]
cluster3=df[df["clusters"]==2]
cluster4=df[df["clusters"]==3]

cluster1

sns.barplot(x="clusters",y="PURCHASES",data=df)
sns.barplot(x="clusters",y="BALANCE",data=df)
sns.barplot(x="clusters",y="INSTALLMENTS_PURCHASES",data=df)
sns.barplot(x="clusters",y="CREDIT_LIMIT",data=df)
sns.barplot(x="clusters",y="PAYMENTS",data=df)
sns.barplot(x="clusters",y="TENURE",data=df)
sns.barplot(x="clusters",y="BALANCE_FREQUENCY",data=df)

df.columns
















# CUSTID: Identification of Credit Card holder 
# BALANCE: Balance amount left in customer's account to make purchases
# BALANCE_FREQUENCY: How frequently the Balance is updated, score between 0 and 1 (1 = frequently updated, 0 = not frequently updated)
# PURCHASES: Amount of purchases made from account
# ONEOFFPURCHASES: Maximum purchase amount done in one-go
# INSTALLMENTS_PURCHASES: Amount of purchase done in installment
# CASH_ADVANCE: Cash in advance given by the user
# PURCHASES_FREQUENCY: How frequently the Purchases are being made, score between 0 and 1 (1 = frequently purchased, 0 = not frequently purchased)
# ONEOFF_PURCHASES_FREQUENCY: How frequently Purchases are happening in one-go (1 = frequently purchased, 0 = not frequently purchased)
# PURCHASES_INSTALLMENTS_FREQUENCY: How frequently purchases in installments are being done (1 = frequently done, 0 = not frequently done)
# CASH_ADVANCE_FREQUENCY: How frequently the cash in advance being paid
# CASH_ADVANCE_TRX: Number of Transactions made with "Cash in Advance"
# PURCHASES_TRX: Number of purchase transactions made
# CREDIT_LIMIT: Limit of Credit Card for user
# PAYMENTS: Amount of Payment done by user
# MINIMUM_PAYMENTS: Minimum amount of payments made by user  
# PRC_FULL_PAYMENT: Percent of full payment paid by user
# TENURE: Tenure of credit card service for user






# The goal of this project is to leverage AI/ ML model to segment customers for launching a specific targeted Ad-campaign. To make it successful, we have to segment them in at-least 3 distinct groups known as "marketing segmentation". It will help to maximize the marketing campaign conversion rate. For example the general four segments are:

# 1. Transactors: Customers who pay least amount of interest and very careful with the money. Generally they have lower balance(USD 104), cash advance (USD 303) and perecnt of full paymenet = 23%
# 2. Revolvers : (Most lucrative sector) use credit card as a loan, generally they have highest balance (USD 5000), cash advance (USD 5000), low purchase frequency, high cash advance frequency (0.5), high cash advance transactions (16).
# 3. VIP/Prime : (This group is specific target to increase credit limit and spend habbit) High credit limit (USD 16K), high percentage of full payment.
# 4. Low Tenure: Low tenure (7 Years), low balance.


# https://www.kaggle.com/arjunbhasin2013/ccdata