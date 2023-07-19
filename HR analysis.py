#!/usr/bin/env python
# coding: utf-8

# In[235]:


import pandas as pd
import numpy as np
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder 
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier as Knn
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import PolynomialFeatures
from sklearn.naive_bayes import GaussianNB
from scipy.stats import ttest_ind
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import ttest_rel
from scipy.stats import chisquare
from scipy.stats import shapiro
from scipy.stats import wilcoxon
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from scipy.stats import ranksums
from sklearn.model_selection import KFold
from sklearn.svm import SVR


# In[9]:


Data = pd.read_csv(r"C:\HRDataset_v14.csv")


# In[10]:


pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 10)
display(Data)


# In[15]:


print(Data.hist(bins = 20,figsize = 
         (15,15)))


# In[28]:


Data["PerformanceScore"]


# In[13]:


corr =  Data.corr("spearman", numeric_only = True)
display(corr.style.background_gradient(cmap='coolwarm'))


# In[17]:


corr =  Data.corr("kendall", numeric_only = True)
display(corr.style.background_gradient(cmap='coolwarm'))


# In[18]:


data = Data
data = data.drop(["DeptID"], axis = 1)
data = data.drop(["GenderID"], axis = 1)
data = data.drop(["Termd"], axis = 1)
data = data.drop(["Zip"], axis = 1)
data = data.drop(["EmpStatusID"], axis = 1)
data = data.drop(["FromDiversityJobFairID"],axis = 1)
data = data.drop(["PositionID"],axis = 1)
data = data.drop(["EmpID"],axis = 1)
data = data.drop(["MarriedID"],axis = 1)
data= data.drop(["MaritalStatusID"],axis = 1)
data= data.drop(["Absences"], axis = 1)
data= data.drop(["ManagerID"], axis = 1)


# In[19]:


len(data)


# In[20]:


display(data.describe())


# In[11]:


print(data.hist(bins=50, figsize=(25, 25)))


# In[1424]:


display(data["PerformanceScore"].hist())


# In[1425]:


display(data["EmpSatisfaction"].hist(bins = 30))


# In[1426]:


display(data["SpecialProjectsCount"].hist(bins = 50))


# In[1427]:


display(data["Salary"].hist(bins = 30))


# In[1428]:


display(data["EngagementSurvey"].hist(bins = 30))


# In[1429]:


display(data["DaysLateLast30"].hist())


# In[1430]:


print(data.corr("kendall")["Salary"])


# In[1431]:


print(data.corr("kendall")["PerfScoreID"])


# In[1432]:


print(data.corr("kendall")["EmpSatisfaction"])


# In[1433]:


print(data.corr("kendall")["SpecialProjectsCount"])


# In[1434]:


print(data.corr("kendall")["DaysLateLast30"])


# In[1435]:


print(data.corr("kendall")["EngagementSurvey"])


# In[1436]:


print("Shapiro test SpecialProjectsCount",shapiro(data["SpecialProjectsCount"]))
print("\nShapiro test Performance score",shapiro(data["PerfScoreID"]))
print("\nShapiro test Salary",shapiro(data["Salary"]))
print("\nShapiro test Employmee Satisfaction",shapiro(data["EmpSatisfaction"]))
print("\nShapiro test 30 Days late",shapiro(data["DaysLateLast30"]))
print("\nShapiro test Engagement Survey",shapiro(data["EngagementSurvey"]))


# In[1437]:


import numpy as np
import seaborn as sns
sns.set_style('whitegrid')
display(sns.kdeplot(np.array(data["Salary"]), bw=0.5))


# In[1438]:


sns.set_style('whitegrid')
display(sns.kdeplot(np.array(data["SpecialProjectsCount"]), bw=0.5))


# In[1439]:


sns.set_style('whitegrid')
display(sns.kdeplot(np.array(data["PerfScoreID"]), bw=0.5))


# In[1440]:


sns.set_style('whitegrid')
display(sns.kdeplot(np.array(data["EmpSatisfaction"]), bw=0.5))


# In[1441]:


sns.set_style('whitegrid')
display(sns.kdeplot(np.array(data["DaysLateLast30"]), bw=0.5))


# In[1442]:


sns.set_style('whitegrid')
display(sns.kdeplot(np.array(data["EngagementSurvey"]), bw=0.5))


# In[1487]:


corr = data.corr("kendall")
display(corr.style.background_gradient(cmap='coolwarm'))


# In[1444]:


corr = data.corr("spearman")
display(corr.style.background_gradient(cmap='coolwarm'))


# In[1489]:


display(data["ManagerID"].hist())


# In[1445]:


plt.scatter(data["EngagementSurvey"], data["PerfScoreID"])
plt.xlabel("Engagment survey")
plt.ylabel('Performance score')
plt.show()


# In[1446]:


plt.scatter(data["Salary"], data["PerfScoreID"])
plt.xlabel("Salary $")
plt.ylabel('Score')
plt.show()


# In[1448]:


plt.boxplot(data["EngagementSurvey"])
plt.show()


# In[1449]:


plt.boxplot(data["Salary"])
plt.show()


# In[ ]:





# # Engagement Survey and Performance score

# In[724]:


plt.scatter(Data["EngagementSurvey"], Data["PerfScoreID"])
plt.xlabel("Engagement survey")
plt.ylabel("Performance Score")
plt.show()


# In[255]:


X = np.array(Data["EngagementSurvey"]).reshape(-1,1)
X = normalization.fit_transform(X)
y = np.array(Data["PerfScoreID"])
kf = KFold(n_splits=5, random_state= 10, shuffle = True )
kf.get_n_splits(X)

print(kf)

for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]


# In[257]:


model= Knn(6)
model.fit(X_train, y_train)
y_predict = model.predict(X_test)
print(metrics.accuracy_score(y_test, y_predict))


# In[153]:


model= LogisticRegression(multi_class='multinomial', solver='lbfgs')
model.fit(X_train, y_train)
y_predict = model.predict(X_test)
print(metrics.accuracy_score(y_test, y_predict))


# In[154]:


model= RandomForestClassifier(n_estimators = 4, max_depth= 30, criterion ="entropy")
model.fit(X_train, y_train)
y_predict = model.predict(X_test)
print(metrics.accuracy_score(y_test, y_predict))


# In[155]:


model= GaussianNB()
model.fit(X_train, y_train)
y_predict = model.predict(X_test)
print(metrics.accuracy_score(y_test, y_predict))


# In[156]:


model= SVC()
model.fit(X_train, y_train)
y_predict = model.predict(X_test)
print(metrics.accuracy_score(y_test, y_predict))


# # Days late last 30 And performance score

# In[725]:


plt.scatter(Data["DaysLateLast30"], Data["PerfScoreID"])
plt.xlabel("Days late last 30 days")
plt.ylabel("Performance Score")
plt.show()


# In[157]:


X = np.array(Data["DaysLateLast30"]).reshape(-1,1)
X = normalization.fit_transform(X)
y = np.array(Data["PerfScoreID"])
kf = KFold(n_splits=5, random_state= 10, shuffle = True )
kf.get_n_splits(X)

print(kf)

for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]


# In[176]:


model= Knn(6)
model.fit(X_train,y_train)
y_predict = model.predict(X_test)
print(metrics.accuracy_score(y_test, y_predict))


# In[169]:


model= LogisticRegression(multi_class='multinomial', solver='lbfgs')
model.fit(X_train, y_train)
y_predict = model.predict(X_test)
print(metrics.accuracy_score(y_test, y_predict))


# In[172]:


model= RandomForestClassifier()
model.fit(X_train, y_train)
y_predict = model.predict(X_test)
print(metrics.accuracy_score(y_test, y_predict))


# In[173]:


model= GaussianNB()
model.fit(X_train, y_train)
y_predict = model.predict(X_test)
print(metrics.accuracy_score(y_test, y_predict))


# In[174]:


model= SVC()
model.fit(X_train, y_train)
y_predict = model.predict(X_test)
print(metrics.accuracy_score(y_test, y_predict))


# # Employment Satisfaction And performance score

# In[811]:


plt.scatter(Data["EmpSatisfaction"], Data["PerfScoreID"])
plt.xlabel("Employee satisfaction")
plt.ylabel("Performance score")
plt.show()


# In[225]:


X = np.array(Data["EmpSatisfaction"]).reshape(-1,1)
X = normalization.fit_transform(X)
y = np.array(Data["PerfScoreID"])
kf = KFold(n_splits=5, shuffle = True, random_state=3 )
kf.get_n_splits(X)

print(kf)

for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]


# In[226]:


model= Knn(5)
model.fit(X_train, y_train)
y_predict = model.predict(X_test)
print(metrics.accuracy_score(y_test, y_predict))


# In[227]:


model= LogisticRegression(multi_class='multinomial', solver='lbfgs')
model.fit(X_train, y_train)
y_predict = model.predict(X_test)
print(metrics.accuracy_score(y_test, y_predict))


# In[228]:


model= RandomForestClassifier(100)
model.fit(X_train, y_train)
y_predict = model.predict(X_test)
print(metrics.accuracy_score(y_test, y_predict))


# In[229]:


model= GaussianNB()
model.fit(X_train, y_train)
y_predict = model.predict(X_test)
print(metrics.accuracy_score(y_test, y_predict))


# In[230]:


model= SVC()
model.fit(X_train, y_train)
y_predict = model.predict(X_test)
print(metrics.accuracy_score(y_test, y_predict))


# # EngagementSurvey And DaysLateLast30

# In[231]:


plt.scatter(Data["DaysLateLast30"], Data["EngagementSurvey"])
plt.xlabel("Days late in last 30 days")
plt.ylabel("Engagement Survey")
plt.show()


# In[232]:


X = np.array(Data["EngagementSurvey"]).reshape(-1,1)
X = normalization.fit_transform(X)
y = np.array(Data["DaysLateLast30"])
kf = KFold(n_splits=5, shuffle = True, random_state=3 )
kf.get_n_splits(X)

print(kf)

for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]


# In[238]:


model= SVR()
model.fit(X_train, y_train)
y_predict = model.predict(X_test)
plt.scatter(X_test, y_predict, color = "red")
plt.scatter(X_test, y_test, color = "blue")
plt.show()
print(mean_squared_error(y_predict, y_test))


# In[240]:


model= LinearRegression()
model.fit(X_train, y_train)
y_predict = model.predict(X_test)
plt.plot(X_test, y_predict, color = "red")
plt.scatter(X_test, y_test, color = "blue")
plt.show()
print(mean_squared_error(y_predict, y_test))


# In[249]:


model= DecisionTreeRegressor(max_depth = 60)
model.fit(X_train, y_train)
y_predict = model.predict(X_test)
plt.scatter(X_test, y_predict, color = "red")
plt.scatter(X_test, y_test, color = "blue")
plt.show()
print(mean_squared_error(y_predict, y_test))


# # Salary and SpecialProjectsCount

# In[26]:


plt.scatter(data["Salary"], data["SpecialProjectsCount"])
plt.xlabel("Salary in $")
plt.ylabel("Special Projects")
plt.show()


# In[264]:


X = np.array(Data["Salary"]).reshape(-1,1)
X = normalization.fit_transform(X)
y = np.array(Data["SpecialProjectsCount"])
kf = KFold(n_splits=5, shuffle = True, random_state=3 )
kf.get_n_splits(X)

print(kf)

for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]


# In[265]:


model= LinearRegression()
model.fit(X_train, y_train)
y_predict = model.predict(X_test)
plt.plot(X_test, y_predict, color = "red")
plt.scatter(X_test, y_test, color = "blue")
plt.show()
print(mean_squared_error(y_predict, y_test))


# In[266]:


model= SVR()
model.fit(X_train, y_train)
y_predict = model.predict(X_test)
plt.scatter(X_test, y_predict, color = "red")
plt.scatter(X_test, y_test, color = "blue")
plt.show()
print(mean_squared_error(y_predict, y_test))


# In[267]:


model= DecisionTreeRegressor()
model.fit(X_train, y_train)
y_predict = model.predict(X_test)
plt.scatter(X_test, y_predict, color = "red")
plt.scatter(X_test, y_test, color = "blue")
plt.show()
print(mean_squared_error(y_predict, y_test))


# In[ ]:





# In[ ]:





# In[336]:


male=[]
female=[]
for i in range(len(Data)):
    if Data["GenderID"][i]== 1:
        male.append(Data.iloc[i,:])
    if Data["GenderID"][i]==0:
        female.append(Data.iloc[i,:])
    
m = pd.DataFrame(male)
f = pd.DataFrame(female)


# In[270]:


m


# In[271]:


f


# In[272]:


m["PerfScoreID"].hist()
plt.xlabel("Performance score")
plt.ylabel("Males")
plt.show()


# In[273]:


f["PerfScoreID"].hist()
plt.xlabel("Performance score")
plt.ylabel("Females")
plt.show()


# In[274]:


f["Salary"].hist(bins = 60)
plt.xlabel("Salary in $")
plt.ylabel("Females")
plt.show()


# In[275]:


m["Salary"].hist(bins = 60)
plt.xlabel("Salary in $")
plt.ylabel("males")
plt.show()


# In[337]:


m = m.drop(["DeptID"], axis = 1)
m = m.drop(["GenderID"], axis = 1)
m = m.drop(["Termd"], axis = 1)
m = m.drop(["Zip"], axis = 1)
m = m.drop(["EmpStatusID"], axis = 1)
m = m.drop(["FromDiversityJobFairID"],axis = 1)
m = m.drop(["PositionID"],axis = 1)
m = m.drop(["EmpID"],axis = 1)
m = m.drop(["MaritalStatusID"],axis = 1)
m = m.drop(["MarriedID"], axis=1)
m = m.drop(["ManagerID"],axis=1)
m = m.drop(["Absences"],axis=1)


# In[338]:


f = f.drop(["DeptID"], axis = 1)
f = f.drop(["GenderID"], axis = 1)
f = f.drop(["Termd"], axis = 1)
f = f.drop(["Zip"], axis = 1)
f = f.drop(["EmpStatusID"], axis = 1)
f = f.drop(["FromDiversityJobFairID"],axis = 1)
f = f.drop(["PositionID"],axis = 1)
f = f.drop(["EmpID"],axis = 1)
f = f.drop(["MaritalStatusID"],axis = 1)
f = f.drop(["MarriedID"], axis=1)
f = f.drop(["Absences"],axis=1)
f = f.drop(["ManagerID"],axis=1)


# In[342]:


f = f.reset_index()


# In[343]:


f = f.drop(["index"],axis = 1)
display(f)


# In[344]:


m =m.reset_index()


# In[345]:


m = m.drop(["index"],axis = 1)
display(m)


# In[346]:


ma = m.corr("spearman")
display(ma.style.background_gradient(cmap='coolwarm'))


# In[347]:


fe = f.corr("spearman")
display(fe.style.background_gradient(cmap='coolwarm'))


# In[1632]:


print(m["EmpSatisfaction"].median())
print(m["SpecialProjectsCount"].mean())
print(m["Salary"].median())
print(m["EmpSatisfaction"].median())
print(m["EmpSatisfaction"].median())
print(m["EmpSatisfaction"].median())


# In[286]:


maleperf = pd.DataFrame(m["PerfScoreID"])
maleSal = pd.DataFrame(m["Salary"])
maleEngag = pd.DataFrame(m["EngagementSurvey"])
maleDaysl = pd.DataFrame(m["DaysLateLast30"])
maleEmpSatis = pd.DataFrame(m["EmpSatisfaction"])
maleProj = pd.DataFrame(m["SpecialProjectsCount"])

x1 = []
x2 = []
x3 = []
x4 = []
x5 = []
x6 = []
for i in range(len(f)-len(m)):
    x1.append({"PerfScoreID":3})
maleperf= maleperf.append(x1, ignore_index = True)

for i in range(len(f)-len(m)):
    x2.append({"Salary":63353.0})
maleSal = maleSal.append(x2, ignore_index = True)

for i in range(len(f)-len(m)):
    x3.append({"EngagementSurvey":4.07})
maleEngag = maleEngag.append(x3, ignore_index = True)

for i in range(len(f)-len(m)):
    x4.append({"DaysLateLast30":0})
maleDaysl = maleDaysl.append(x4, ignore_index = True)

for i in range(len(f)-len(m)):
    x5.append({"EmpSatisfaction":4.0})
maleEmpSatis = maleEmpSatis.append(x5, ignore_index = True)

for i in range(len(f)-len(m)):
    x6.append({"SpecialProjectsCount":0.0})
maleProj = maleProj.append(x6, ignore_index = True)


# In[352]:


frames = [maleperf, maleSal, maleDaysl, maleEmpSatis, maleProj, maleEngag ]
male = pd.concat(frames,axis =1)


# In[365]:


display(male.hist(figsize = (15,15)))


# In[366]:


display(f.hist(figsize = (15,15)))


# In[300]:


print(ranksums(male["EngagementSurvey"],f["EngagementSurvey"]))


# In[301]:


print(ranksums(male["Salary"],f["Salary"]))


# In[302]:


print(ranksums(male["SpecialProjectsCount"],f["SpecialProjectsCount"]))


# In[303]:


print(ranksums(male["EmpSatisfaction"],f["EmpSatisfaction"]))


# In[304]:


print(ranksums(male["PerfScoreID"],f["PerfScoreID"]))


# In[1711]:


print(All.corr("spearman"))


# In[ ]:




