from sklearn.model_selection import train_test_split
from sklearn . preprocessing import StandardScaler
from sklearn . metrics import confusion_matrix
from sklearn . neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import math
pd.options.mode.chained_assignment = None

goog_path = os.path.abspath('GOOG_weekly_return_volatility.csv')
df_goog = pd.read_csv(goog_path)
df_googvol = df_goog[df_goog.Year.isin([2019])]
df_googvol_2yrs = df_goog[df_goog.Year.isin([2019,2020])]
df_googvol_2020 = df_goog[df_goog.Year.isin([2020])]

print('####################Q0#####################')
print('********Computation with P = 2,Euclidean ************')
error_rate = []
Pred_lst  =[]
Y_test_lst = []
k_lst = [3,5,7,9,11]

for k in range (3 ,13 ,2):
    X = df_googvol [["mean_return", "volatility"]]
    y = df_googvol["Label"]
    scaler = StandardScaler (). fit (X)
    X = scaler . transform (X)
    X_train ,X_test , Y_train , Y_test = train_test_split (X,y,test_size =0.5)
    knn_classifier = KNeighborsClassifier ( n_neighbors =k)
    knn_classifier.fit(X_train , Y_train )
    pred_k = knn_classifier.predict(X_test)
    Pred_lst.append(pred_k)
    error_rate.append(np.mean(pred_k == Y_test))
    Y_test_lst.append(Y_test)
    print("The Accuracy is {} when k is {} for 2019 with p =2".format(np.mean(pred_k == Y_test), k )) 
print("All Accuracies for 2019 with p =2 are  : ", error_rate)
plt.figure ( figsize =(10 ,4))
ax = plt. gca ()
ax.xaxis.set_major_locator ( plt.MaxNLocator ( integer = True ))
plt.plot ( range (3 ,13 ,2) , error_rate , color ='red', linestyle ='dotted',
marker ='o', markerfacecolor ='black', markersize =10)
plt.title ('Accuracy vs. k for p=2')
plt.xlabel ('number of neighbors : k')
plt.ylabel ('Accuracy')


max_value = max(error_rate)
max_index = error_rate.index(max_value)
kvalp2 = k_lst[max_index]
print('Optimal value for k in 2019 is : ', k_lst[max_index])

## Q0. Part 2 - Predicting labels for 2020 using optimal k from 2019 

df_googvol_2yrs_test = pd.DataFrame (
{"mean_return": df_googvol_2yrs.iloc[:,2].tolist(),
"volatility":df_googvol_2yrs.iloc[:,3].tolist(),
"Label":df_googvol_2yrs.iloc[:,4].tolist()},
columns = ["mean_return","volatility", "Label"])
X = df_googvol_2yrs_test [["mean_return", "volatility"]]
y = df_googvol_2yrs_test["Label"]
scaler = StandardScaler (). fit (X)
X = scaler . transform (X)
X_train ,X_test , Y_train , Y_test = train_test_split (X,y,test_size =0.5, shuffle = False)
knn_classifier = KNeighborsClassifier ( n_neighbors = k_lst[max_index])
knn_classifier.fit(X_train , Y_train )
pred_k_2020 = knn_classifier.predict(X_test)
print("pred_k_2020", pred_k_2020)
error_rate.append(np.mean(pred_k_2020 == Y_test))
Acc_per_p2 = round(np.mean(pred_k_2020 == Y_test)*100)
print("The Accuracy is {} when k is {}  for year 2020 with p =2 ".format(np.mean(pred_k_2020 == Y_test), k_lst[max_index] )) 


###Q0( part 3,4)  -> Confusion matrix , TNR and TPR for year 2 ie 2020 ####
cf_1 = confusion_matrix( Y_test , pred_k_2020 )
print("Confusion matrix for year 2020  for k {} is {} ".format(k_lst[max_index], cf_1))
tpr = cf_1[1][1]/(cf_1[1][1] + cf_1[1][0])
tnr = cf_1[0][0]/(cf_1[0][0] + cf_1[0][1])
print("TPR  for year 2020 is {}  and TNR for year 2020 is {} with p =2".format( tpr, tnr))

print('***** Labels buy and hold and trading Strategy *****')

googd_path = os.path.abspath('GOOG_weekly_return_volatility_detailed.csv')

df_googvold = pd.read_csv(googd_path,parse_dates=["Date"],dayfirst=True).drop(['High','Low','Open','Close','Volume', 'mean_return','volatility'], axis=1).sort_values('Date')
df_googvold['Open'] = df_googvold['Adj Close'].shift(1)
df_googvold['Close'] = df_googvold['Adj Close']
df_googvold = df_googvold.drop(['Adj Close'], axis = 1)

df_googvold = df_googvold[df_googvold.Year.isin([2020])]
df_goog = df_googvold.groupby(['Year','Week_Number']).agg({'Date': ['min','max']}).reset_index()
df_goog.columns = ['Year','Week_Number','OpenDate','CloseDate']
df_goog = (df_goog.merge(df_googvold[['Date','Open']], left_on = 'OpenDate', right_on = 'Date')
      .drop('Date',axis=1)
      .merge(df_googvold[['Date','Close']], left_on = 'CloseDate', right_on = 'Date')
      .drop('Date',axis=1))

df_goog = df_goog.merge(df_googvol_2yrs[['Week_Number','Year']],how='left',left_on=['Week_Number','Year'],right_on = ['Week_Number','Year'])

df_goog['Label'] = pred_k_2020
df_goog['NexLabel'] = df_goog['Label'].shift(-1)

cap = 100 + 100*(df_goog.loc[52,'Close'] - df_goog.loc[0,'Open'])/df_goog.loc[0,'Open']
print("GOOG buy-hold  cap for 2020 : {}".format(cap))

cap  = 100
op = 0
for index, row in df_goog.iterrows():
    if row[6] == 1 and op == 0:
        op = row[4]
    if row[6] == 1 and row[7] == 0:
        cap = cap + cap * ((row[5] - op)/op)
        op = 0

strategy_p2 = round(cap)
print("GOOG trading startegy based on label cap for 2020 with p =2: {}".format(cap))



print('####################Q1#####################')
error_rate = []
Pred_lst  =[]
Y_test_lst = []
k_lst = [3,5,7,9,11]

for k in range (3 ,13 ,2):
    X = df_googvol [["mean_return", "volatility"]]
    y = df_googvol["Label"]
    scaler = StandardScaler (). fit (X)
    X = scaler . transform (X)
    X_train ,X_test , Y_train , Y_test = train_test_split (X,y,test_size =0.5)
    knn_classifier = KNeighborsClassifier ( n_neighbors =k , p=1)
    knn_classifier.fit(X_train , Y_train )
    pred_k = knn_classifier.predict(X_test)
    Pred_lst.append(pred_k)
    error_rate.append(np.mean(pred_k == Y_test))
    Y_test_lst.append(Y_test)
    print("The Accuracy is {} when k is {} for 2019 with p =1 ".format(np.mean(pred_k == Y_test), k )) 
print("All Accuracies for 2019 with p =1 are  : ", error_rate)
plt.figure ( figsize =(10 ,4))
ax = plt. gca ()
ax.xaxis.set_major_locator ( plt.MaxNLocator ( integer = True ))
plt.plot ( range (3 ,13 ,2) , error_rate , color ='red', linestyle ='dotted',
marker ='o', markerfacecolor ='black', markersize =10)
plt.title ('Accuracy vs. k for p=1')
plt.xlabel ('number of neighbors : k')
plt.ylabel ('Accuracy')


max_value = max(error_rate)
max_index = error_rate.index(max_value)
kvalp1 = k_lst[max_index]
print('Optimal value for k in 2019 is : ', k_lst[max_index])

## Q1. Part 2 - Predicting labels for 2020 using optimal k from 2019 

df_googvol_2yrs_test = pd.DataFrame (
{"mean_return": df_googvol_2yrs.iloc[:,2].tolist(),
"volatility":df_googvol_2yrs.iloc[:,3].tolist(),
"Label":df_googvol_2yrs.iloc[:,4].tolist()},
columns = ["mean_return","volatility", "Label"])
X = df_googvol_2yrs_test [["mean_return", "volatility"]]
y = df_googvol_2yrs_test["Label"]
scaler = StandardScaler (). fit (X)
X = scaler . transform (X)
X_train ,X_test , Y_train , Y_test = train_test_split (X,y,test_size =0.5, shuffle = False)
knn_classifier = KNeighborsClassifier ( n_neighbors = k_lst[max_index],p=1)
knn_classifier.fit(X_train , Y_train )
pred_k_2020P1 = knn_classifier.predict(X_test)
print("pred_k_2020P1", pred_k_2020P1)
error_rate.append(np.mean(pred_k_2020P1 == Y_test))
Acc_per_p1 = round(np.mean(pred_k_2020P1 == Y_test)*100)
print("The Accuracy is {} when k is {}  for year 2020 with p =1 ".format(np.mean(pred_k_2020P1 == Y_test), k_lst[max_index] )) 


###Q1( part 3,4)  -> Confusion matrix , TNR and TPR for year 2 ie 2020 ####
cf_1 = confusion_matrix( Y_test , pred_k_2020P1 )
print("Confusion matrix for year 2020  for k {} is {} ".format(k_lst[max_index], cf_1))
tpr = cf_1[1][1]/(cf_1[1][1] + cf_1[1][0])
tnr = cf_1[0][0]/(cf_1[0][0] + cf_1[0][1])
print("TPR  for year 2020 is {}  and TNR for year 2020 is {} with p =1 ".format( tpr, tnr))

print('***** Labels buy and hold and trading Strategy *****')

googd_path = os.path.abspath('GOOG_weekly_return_volatility_detailed.csv')

df_googvold = pd.read_csv(googd_path,parse_dates=["Date"],dayfirst=True).drop(['High','Low','Open','Close','Volume', 'mean_return','volatility'], axis=1).sort_values('Date')
df_googvold['Open'] = df_googvold['Adj Close'].shift(1)
df_googvold['Close'] = df_googvold['Adj Close']
df_googvold = df_googvold.drop(['Adj Close'], axis = 1)

df_googvold = df_googvold[df_googvold.Year.isin([2020])]
df_goog = df_googvold.groupby(['Year','Week_Number']).agg({'Date': ['min','max']}).reset_index()
df_goog.columns = ['Year','Week_Number','OpenDate','CloseDate']
df_goog = (df_goog.merge(df_googvold[['Date','Open']], left_on = 'OpenDate', right_on = 'Date')
      .drop('Date',axis=1)
      .merge(df_googvold[['Date','Close']], left_on = 'CloseDate', right_on = 'Date')
      .drop('Date',axis=1))

df_goog = df_goog.merge(df_googvol_2yrs[['Week_Number','Year']],how='left',left_on=['Week_Number','Year'],right_on = ['Week_Number','Year'])


df_goog['Label'] = pred_k_2020P1
df_goog['NexLabel'] = df_goog['Label'].shift(-1)


cap = 100 + 100*(df_goog.loc[52,'Close'] - df_goog.loc[0,'Open'])/df_goog.loc[0,'Open']
print("GOOG buy-hold  cap for 2020 : {}".format(cap))

cap  = 100
op = 0
for index, row in df_goog.iterrows():
    if row[6] == 1 and op == 0:
        op = row[4]
    if row[6] == 1 and row[7] == 0:
        cap = cap + cap * ((row[5] - op)/op)
        op = 0

strategy_p1 = round(cap)
print("GOOG trading strategy based on label cap for 2020 with p =1 is : {}".format(cap))


print('####################Q2#####################')


def minkowski_p(a,b,p):
    return np.linalg.norm(a-b, ord=p)

error_rate = []
Pred_lst  =[]
Y_test_lst = []
k_lst = [3,5,7,9,11]

for k in range (3 ,13 ,2):
    X = df_googvol [["mean_return", "volatility"]]
    y = df_googvol["Label"]
    scaler = StandardScaler (). fit (X)
    X = scaler . transform (X)
    X_train ,X_test , Y_train , Y_test = train_test_split (X,y,test_size =0.5)
    p = 1.5
    knn_Minkowski_p = KNeighborsClassifier(n_neighbors=k, metric = lambda a,b: minkowski_p(a,b,p) ) 
    knn_Minkowski_p.fit(X_train , Y_train )
    pred_k = knn_classifier.predict(X_test)
    Pred_lst.append(pred_k)
    error_rate.append(np.mean(pred_k == Y_test))
    Y_test_lst.append(Y_test)
    print("The Accuracy is {} when k is {} for 2019 with p =1.5 ".format(np.mean(pred_k == Y_test), k )) 
print("All Accuracies for 2019 with p =1.5 are  : ", error_rate)
plt.figure ( figsize =(10 ,4))
ax = plt. gca ()
ax.xaxis.set_major_locator ( plt.MaxNLocator ( integer = True ))
plt.plot ( range (3 ,13 ,2) , error_rate , color ='red', linestyle ='dotted',
marker ='o', markerfacecolor ='black', markersize =10)
plt.title ('Accuracy vs. k for p = 1.5')
plt.xlabel ('number of neighbors : k')
plt.ylabel ('Accuracy')


max_value = max(error_rate)
max_index = error_rate.index(max_value)
kvalp105 = k_lst[max_index]
print('Optimal value for k in 2019 is : ', k_lst[max_index])

## Q2. Part 2 - Predicting labels for 2020 using optimal k from 2019 

df_googvol_2yrs_test = pd.DataFrame (
{"mean_return": df_googvol_2yrs.iloc[:,2].tolist(),
"volatility":df_googvol_2yrs.iloc[:,3].tolist(),
"Label":df_googvol_2yrs.iloc[:,4].tolist()},
columns = ["mean_return","volatility", "Label"])
X = df_googvol_2yrs_test [["mean_return", "volatility"]]
y = df_googvol_2yrs_test["Label"]
scaler = StandardScaler (). fit (X)
X = scaler . transform (X)
X_train ,X_test , Y_train , Y_test = train_test_split (X,y,test_size =0.5, shuffle = False)
p = 1.5
knn_Minkowski_p = KNeighborsClassifier(n_neighbors= k_lst[max_index], metric = lambda a,b: minkowski_p(a,b,p) ) 
knn_Minkowski_p.fit(X_train , Y_train )
pred_k_2020P105 = knn_classifier.predict(X_test)
print("pred_k_2020P105", pred_k_2020P105)
error_rate.append(np.mean(pred_k_2020P105 == Y_test))
Acc_per_p105 = round(np.mean(pred_k_2020P105 == Y_test)*100)
print("The Accuracy is {} when k is {}  for year 2020 with p =1.5 ".format(np.mean(pred_k_2020P105 == Y_test), k_lst[max_index] )) 


###Q2( part 3,4)  -> Confusion matrix , TNR and TPR for year 2 ie 2020 ####
cf_1 = confusion_matrix( Y_test , pred_k_2020P105 )
print("Confusion matrix for year 2020  for k {} is {} ".format(k_lst[max_index], cf_1))
tpr = cf_1[1][1]/(cf_1[1][1] + cf_1[1][0])
tnr = cf_1[0][0]/(cf_1[0][0] + cf_1[0][1])
print("TPR  for year 2020 is {}  and TNR for year 2020 is {} with p =1.5 ".format( tpr, tnr))

print('***** Labels buy and hold and trading Strategy *****')

googd_path = os.path.abspath('GOOG_weekly_return_volatility_detailed.csv')

df_googvold = pd.read_csv(googd_path,parse_dates=["Date"],dayfirst=True).drop(['High','Low','Open','Close','Volume', 'mean_return','volatility'], axis=1).sort_values('Date')
df_googvold['Open'] = df_googvold['Adj Close'].shift(1)
df_googvold['Close'] = df_googvold['Adj Close']
df_googvold = df_googvold.drop(['Adj Close'], axis = 1)

df_googvold = df_googvold[df_googvold.Year.isin([2020])]
df_goog = df_googvold.groupby(['Year','Week_Number']).agg({'Date': ['min','max']}).reset_index()
df_goog.columns = ['Year','Week_Number','OpenDate','CloseDate']
df_goog = (df_goog.merge(df_googvold[['Date','Open']], left_on = 'OpenDate', right_on = 'Date')
      .drop('Date',axis=1)
      .merge(df_googvold[['Date','Close']], left_on = 'CloseDate', right_on = 'Date')
      .drop('Date',axis=1))

df_goog = df_goog.merge(df_googvol_2yrs[['Week_Number','Year']],how='left',left_on=['Week_Number','Year'],right_on = ['Week_Number','Year'])


df_goog['Label'] = pred_k_2020P105
df_goog['NexLabel'] = df_goog['Label'].shift(-1)


cap = 100 + 100*(df_goog.loc[52,'Close'] - df_goog.loc[0,'Open'])/df_goog.loc[0,'Open']
print("GOOG buy-hold  cap for 2020 : {}".format(cap))

cap  = 100
op = 0
for index, row in df_goog.iterrows():
    if row[6] == 1 and op == 0:
        op = row[4]
    if row[6] == 1 and row[7] == 0:
        cap = cap + cap * ((row[5] - op)/op)
        op = 0

strategy_p105 = round(cap)
print("GOOG trading strategy based on label cap for 2020 with p =1.5 is : {}".format(cap))


print('####################Q3#####################')

  
df_means = df_googvol.groupby('Label').mean() 
df_means = df_means.drop('Year',axis=1)
df_means = df_means.drop('Week_Number',axis=1)
# print(df_means)
Rx1= [df_means.iloc[0,0].tolist()]*53
Ry1= [df_means.iloc[0,1].tolist()]*53
Gx1= [df_means.iloc[1,0].tolist()]*53
Gy1= [df_means.iloc[1,1].tolist()]*53
# print(Rx1,Ry1,Gx1,Gy1)
print("Centre for red centroid is x = {} and y = {}: ".format(df_means.iloc[0,0],df_means.iloc[0,1]))
print("Centre for green centroid is x = {} and y = {}: ".format(df_means.iloc[1,0],df_means.iloc[1,1]))

x_lst = df_googvol.mean_return.tolist() 
y_lst = df_googvol.volatility.tolist()         
df_temp = pd.DataFrame(list(zip(x_lst, y_lst,Rx1, Ry1,Gx1, Gy1)),
                columns =['x2', 'y2', 'Rx1' , 'Ry1','Gx1' , 'Gy1'])

r_list = ((df_temp['x2']- df_temp['Rx1'])**2 + (df_temp['y2']- df_temp['Ry1'])**2).pow(.5)
g_list = ((df_temp['x2']- df_temp['Gx1'])**2 + (df_temp['y2']- df_temp['Gy1'])**2).pow(.5)

df_temp['r_list'] = r_list
df_temp['g_list'] = g_list

df_temp['Label'] = np.where(df_temp['r_list'] > df_temp['g_list'] , 0, 1)

print("The avg distance of Red centroid is : {}".format(r_list.mean()))
print("The avg distance of Green centroid is : {}".format(g_list.mean()))
print("The median distance of Red centroid is : {}".format(r_list.median()))
print("The median distance of Green centroid is : {}".format(g_list.median()))


############# Using this algorithm, computing the predicted labels for 2020 #######


df_means = df_googvol_2020.groupby('Label').mean() 
df_googvol_True_label = df_googvol_2020.Label.tolist()
df_means = df_means.drop('Year',axis=1)
df_means = df_means.drop('Week_Number',axis=1)
# print(df_means)
Rx1= [df_means.iloc[0,0].tolist()]*53
Ry1= [df_means.iloc[0,1].tolist()]*53
Gx1= [df_means.iloc[1,0].tolist()]*53
Gy1= [df_means.iloc[1,1].tolist()]*53
x_lst = df_googvol_2020.mean_return.tolist() 
y_lst = df_googvol_2020.volatility.tolist()         
df_temp = pd.DataFrame(list(zip(x_lst, y_lst,Rx1, Ry1,Gx1, Gy1)),
                columns =['x2', 'y2', 'Rx1' , 'Ry1','Gx1' , 'Gy1'])
r_list = ((df_temp['x2']- df_temp['Rx1'])**2 + (df_temp['y2']- df_temp['Ry1'])**2).pow(.5)
g_list = ((df_temp['x2']- df_temp['Gx1'])**2 + (df_temp['y2']- df_temp['Gy1'])**2).pow(.5)
df_temp['r_list'] = r_list
df_temp['g_list'] = g_list
df_temp['Label'] = np.where(df_temp['r_list'] > df_temp['g_list'] , 0, 1)
pred_label = df_temp.Label.tolist()


###Q3( part 2)  -> Confusion matrix , TNR and TPR for year 2 ie 2020 ####

cf_1 = confusion_matrix( df_googvol_True_label , pred_label )
print("Confusion matrix for year 2020 is {} ".format(cf_1))
tpr = cf_1[1][1]/(cf_1[1][1] + cf_1[1][0])
tnr = cf_1[0][0]/(cf_1[0][0] + cf_1[0][1])
Acc = round((cf_1[0][0]+cf_1[1][1])/(cf_1[0][0]+cf_1[1][1]+cf_1[1][0]+cf_1[0][1])*100)
print("TPR  for year 2020 is {}  and TNR for year 2020 is {} ".format( tpr, tnr))

true_pred = pd.DataFrame(list(zip(df_googvol_True_label, pred_label)),
                columns =['true_label', 'pred_label'])
#print(true_pred)

###Q3( part 3) 
print('***** Labels buy and hold and trading Strategy *****')

googd_path = os.path.abspath('GOOG_weekly_return_volatility_detailed.csv')

df_googvold = pd.read_csv(googd_path,parse_dates=["Date"],dayfirst=True).drop(['High','Low','Open','Close','Volume', 'mean_return','volatility'], axis=1).sort_values('Date')
df_googvold['Open'] = df_googvold['Adj Close'].shift(1)
df_googvold['Close'] = df_googvold['Adj Close']
df_googvold = df_googvold.drop(['Adj Close'], axis = 1)

df_googvold = df_googvold[df_googvold.Year.isin([2020])]
df_goog = df_googvold.groupby(['Year','Week_Number']).agg({'Date': ['min','max']}).reset_index()
df_goog.columns = ['Year','Week_Number','OpenDate','CloseDate']
df_goog = (df_goog.merge(df_googvold[['Date','Open']], left_on = 'OpenDate', right_on = 'Date')
      .drop('Date',axis=1)
      .merge(df_googvold[['Date','Close']], left_on = 'CloseDate', right_on = 'Date')
      .drop('Date',axis=1))

df_goog = df_goog.merge(df_googvol_2yrs[['Week_Number','Year']],how='left',left_on=['Week_Number','Year'],right_on = ['Week_Number','Year'])

df_goog['Label'] = pred_label
df_goog['NexLabel'] = df_goog['Label'].shift(-1)


cap = 100 + 100*(df_goog.loc[52,'Close'] - df_goog.loc[0,'Open'])/df_goog.loc[0,'Open']
print("GOOG buy-hold  cap for 2020 : {}".format(cap))

cap  = 100
op = 0
for index, row in df_goog.iterrows():
    if row[6] == 1 and op == 0:
        op = row[4]
    if row[6] == 1 and row[7] == 0:
        cap = cap + cap * ((row[5] - op)/op)
        op = 0

strategy_centroid = round(cap)
print("GOOG trading strategy based on label cap for 2020 by centroid algo is : {}".format(cap))

print('####################Q4#####################')
error_rate = []
Pred_lst  =[]
Y_test_lst = []
k_lst = [3,5,7,9,11]
df_x  = df_googvol['mean_return']
df_y  = df_googvol['volatility']
tlabel = df_googvol['Label']
xsqr = ((df_x)**2).tolist()
ysqr =  ((df_y)**2).tolist()
xyr2 = (((df_x)* (df_x))*math.sqrt(2) ).tolist()
df_temp = pd.DataFrame(list(zip(xsqr, ysqr, xyr2,tlabel)),
                columns =['mean_return', 'volatility', 'xyrt2','Label' ])

for k in range (3 ,13 ,2):
  
    X = df_temp [["mean_return", "volatility", "xyrt2"]]
    y = df_temp["Label"]
    scaler = StandardScaler (). fit (X)
    X = scaler . transform (X)
    X_train ,X_test , Y_train , Y_test = train_test_split (X,y,test_size =0.5)
    knn_classifier = KNeighborsClassifier ( n_neighbors =k)
    knn_classifier.fit(X_train , Y_train )
    pred_k = knn_classifier.predict(X_test)
    Pred_lst.append(pred_k)
    error_rate.append(np.mean(pred_k == Y_test))
    Y_test_lst.append(Y_test)
    print("The Accuracy is {} when k is {} for 2019 for Domain Transformation algo".format(np.mean(pred_k == Y_test), k )) 
print("All Accuracies for 2019 for Domain Transformation are  : ", error_rate)
plt.figure ( figsize =(10 ,4))
ax = plt. gca ()
ax.xaxis.set_major_locator ( plt.MaxNLocator ( integer = True ))
plt.plot ( range (3 ,13 ,2) , error_rate , color ='red', linestyle ='dotted',
marker ='o', markerfacecolor ='black', markersize =10)
plt.title ('Accuracy vs. k for Domain Transformation ')
plt.xlabel ('number of neighbors : k')
plt.ylabel ('Accuracy')


max_value = max(error_rate)
max_index = error_rate.index(max_value)
kvaldt = k_lst[max_index]
print('Optimal value for k in 2019 is : ', k_lst[max_index])


## Q4. Part 2 - Predicting labels for 2020 using optimal k from 2019 
df_x  = df_googvol_2yrs['mean_return']
df_y  = df_googvol_2yrs['volatility']
tlabel = df_googvol_2yrs['Label']
xsqr = ((df_x)**2).tolist()
ysqr =  ((df_y)**2).tolist()
xyr2 = (((df_x)* (df_x))*math.sqrt(2) ).tolist()
df_temp = pd.DataFrame(list(zip(xsqr, ysqr, xyr2,tlabel)),
                columns =['mean_return', 'volatility', 'xyrt2','Label' ])

X = df_temp [["mean_return", "volatility", "xyrt2"]]
y = df_temp["Label"]
scaler = StandardScaler (). fit (X)
X = scaler . transform (X)
X_train ,X_test , Y_train , Y_test = train_test_split (X,y,test_size =0.5, shuffle = False)
knn_classifier = KNeighborsClassifier ( n_neighbors = k_lst[max_index])
knn_classifier.fit(X_train , Y_train )
pred_k_2020 = knn_classifier.predict(X_test)
print("pred_k_2020", pred_k_2020)
error_rate.append(np.mean(pred_k_2020 == Y_test))
Acc_per_dt = round(np.mean(pred_k_2020 == Y_test)*100)
print("The Accuracy is {} when k is {}  for year 2020 for Domain Transformation ".format(np.mean(pred_k_2020 == Y_test), k_lst[max_index] )) 


###Q4( part 3,5)  -> Confusion matrix , TNR and TPR for year 2 ie 2020 ####
cf_1 = confusion_matrix( Y_test , pred_k_2020 )
print("Confusion matrix for year 2020  for k {} is {} ".format(k_lst[max_index], cf_1))
tpr = cf_1[1][1]/(cf_1[1][1] + cf_1[1][0])
tnr = cf_1[0][0]/(cf_1[0][0] + cf_1[0][1])
print("TPR  for year 2020 is {}  and TNR for year 2020 is {} with Domain Transformation".format( tpr, tnr))

print('***** Labels buy and hold and trading Strategy *****')

googd_path = os.path.abspath('GOOG_weekly_return_volatility_detailed.csv')

df_googvold = pd.read_csv(googd_path,parse_dates=["Date"],dayfirst=True).drop(['High','Low','Open','Close','Volume', 'mean_return','volatility'], axis=1).sort_values('Date')
df_googvold['Open'] = df_googvold['Adj Close'].shift(1)
df_googvold['Close'] = df_googvold['Adj Close']
df_googvold = df_googvold.drop(['Adj Close'], axis = 1)

df_googvold = df_googvold[df_googvold.Year.isin([2020])]
df_goog = df_googvold.groupby(['Year','Week_Number']).agg({'Date': ['min','max']}).reset_index()
df_goog.columns = ['Year','Week_Number','OpenDate','CloseDate']
df_goog = (df_goog.merge(df_googvold[['Date','Open']], left_on = 'OpenDate', right_on = 'Date')
      .drop('Date',axis=1)
      .merge(df_googvold[['Date','Close']], left_on = 'CloseDate', right_on = 'Date')
      .drop('Date',axis=1))

df_goog = df_goog.merge(df_googvol_2yrs[['Week_Number','Year']],how='left',left_on=['Week_Number','Year'],right_on = ['Week_Number','Year'])

df_goog['Label'] = pred_k_2020
df_goog['NexLabel'] = df_goog['Label'].shift(-1)

cap_bnhold = 100 + 100*(df_goog.loc[52,'Close'] - df_goog.loc[0,'Open'])/df_goog.loc[0,'Open']
print("GOOG buy-hold  cap for 2020 : {}".format(cap_bnhold))

cap  = 100
op = 0
for index, row in df_goog.iterrows():
    if row[6] == 1 and op == 0:
        op = row[4]
    if row[6] == 1 and row[7] == 0:
        cap = cap + cap * ((row[5] - op)/op)
        op = 0

strategy_dt = round(cap)
print("GOOG trading startegy based on label cap for 2020 for Domain Transformation is : {}".format(cap))

print('####################Q5#####################')
k_lst = [3,5,7,9,11]
error_rate = []  
for val in k_lst :
    desc = ['mean_return', 'volatility','Label']
    X = df_googvol[desc].values
    Y = df_googvol['Label'].values
    X_train ,X_test , Y_train , Y_test = train_test_split (X,Y,test_size =0.5)
    predictlabel3 = []
    Countfinal = 0        
    df_train_2019 = pd.DataFrame(X_train, columns = ['mean_return', 'volatility','Label'])
            
    def labelpred(inpdataframe):
        predictlabel1,predictlabel2 = [],[]
        for i in (inpdataframe.index):
            primarydistance =[]
            counter=0
            counter1 = 0
            counter2 =0
            counter3=0           
            for j in range(0,len(X_train)):      
                result = ((inpdataframe.loc[i]['mean_return']-X_train[j][0])**2 + (inpdataframe.loc[i]['volatility']-X_train[j][1])**2)*0.5
                primarydistance.append(result)
            series = pd.Series(primarydistance)
            df_train_2019['dist']= series.values
            seriessort = (df_train_2019.sort_values('dist'))
            kval= (seriessort[1:(val+1)]['Label'])
            for k in (kval.index):
                if(kval[k] == 0):
                    counter = counter + 1
                else:
                    counter1 = counter1 + 1
            if(counter>counter1):
                predictlabel1.append(0)
            else:
                predictlabel1.append(1)
        for k in range(0,len(predictlabel1)):
            if(predictlabel1[k] == 0):
                counter2 = counter2 + 1
            else:
                counter3 = counter3 + 1
        if(counter2>counter3):
            return(0)
        else:
            return(1)           
     
    def distancewithTrain(X_testX,X_testY):
        distance = []

        for a in range(0,len(X_train)): 
            pointdistance = ((X_testX-X_train[a][0])**2 + (X_testY-X_train[a][1])**2)*0.5
            distance.append(pointdistance)
        series = pd.Series(distance)
        df_train_2019['dist']= series.values
        seriessort = (df_train_2019.sort_values('dist'))
        seriessorted = seriessort[0:val]
        retval = labelpred(seriessorted)
        return(retval)    
    
    for i in range(0,len(X_test)):
        distance_label3 = distancewithTrain(X_test[i][0],X_test[i][1])
        predictlabel3.append(distance_label3)        
    
    for i in range(0,len(predictlabel3)):
        if(predictlabel3[i]==Y_test[i]):
            Countfinal = Countfinal + 1
    print("The Accuracy is {} when k is {} for 2019 with K-Predicted Neighbors Algo".format(round(Countfinal/len(predictlabel3),2),val))         
    error_rate.append(Countfinal/len(predictlabel3))

print("All Accuracies for 2019 with KNN are  : ", error_rate)
plt.figure ( figsize =(10 ,4))
ax = plt. gca ()
ax.xaxis.set_major_locator ( plt.MaxNLocator ( integer = True ))
plt.plot ( range (3 ,13 ,2) , error_rate , color ='red', linestyle ='dotted',
marker ='o', markerfacecolor ='black', markersize =10)
plt.title ('Accuracy vs. k for k NEAREST NEIGHBORS')
plt.xlabel ('number of neighbors : k')
plt.ylabel ('Accuracy')

max_value = max(error_rate)
print(max_value)
max_index = error_rate.index(max_value)
print(max_index)
kvalknn = k_lst[max_index]
print('Optimal value for k in 2019 is : ', kvalknn)

########### Q5 Part 2 - Predicting labels for 2020 using optimal k from 2019 #####################3
k_lst = [kvalknn]
for val in k_lst :
    desc = ['mean_return', 'volatility','Label']
    X = df_googvol_2yrs[desc].values
    Y = df_googvol_2yrs['Label'].values
    X_train ,X_test , Y_train , Y_test = train_test_split (X,Y,test_size =0.5, shuffle = False)
    predictlabel3 = []
    Countfinal = 0        
    df_train_2019 = pd.DataFrame(X_train, columns = ['mean_return', 'volatility','Label'])
            
    def labelpred(inpdataframe):
        predictlabel1,predictlabel2 = [],[]
        for i in (inpdataframe.index):
            primarydistance =[]
            counter=0
            counter1 = 0
            counter2 =0
            counter3=0           
            for j in range(0,len(X_train)):      
                result = ((inpdataframe.loc[i]['mean_return']-X_train[j][0])**2 + (inpdataframe.loc[i]['volatility']-X_train[j][1])**2)*0.5
                primarydistance.append(result)
            series = pd.Series(primarydistance)
            df_train_2019['dist']= series.values
            seriessort = (df_train_2019.sort_values('dist'))
            kval= (seriessort[1:(val+1)]['Label'])
            for k in (kval.index):
                if(kval[k] == 0):
                    counter = counter + 1
                else:
                    counter1 = counter1 + 1
            if(counter>counter1):
                predictlabel1.append(0)
            else:
                predictlabel1.append(1)
        for k in range(0,len(predictlabel1)):
            if(predictlabel1[k] == 0):
                counter2 = counter2 + 1
            else:
                counter3 = counter3 + 1
        if(counter2>counter3):
            return(0)
        else:
            return(1)           
     
    def distancewithTrain(X_testX,X_testY):
        distance = []

        for a in range(0,len(X_train)): 
            pointdistance = ((X_testX-X_train[a][0])**2 + (X_testY-X_train[a][1])**2)*0.5
            distance.append(pointdistance)
        series = pd.Series(distance)
        df_train_2019['dist']= series.values
        seriessort = (df_train_2019.sort_values('dist'))
        seriessorted = seriessort[0:val]
        retval = labelpred(seriessorted)
        return(retval)    
    
    for i in range(0,len(X_test)):
        distance_label3 = distancewithTrain(X_test[i][0],X_test[i][1])
        predictlabel3.append(distance_label3)        
    
    for i in range(0,len(predictlabel3)):
        if(predictlabel3[i]==Y_test[i]):
            Countfinal = Countfinal + 1
    Acc_KNN = round(round(Countfinal/len(predictlabel3),2)*100)
    print("The Accuracy is {} when k is {} for 2020 with K-Predicted Neighbors Algo".format(round(Countfinal/len(predictlabel3),2),val))         
    error_rate.append(Countfinal/len(predictlabel3))

print("pred_k_2020", predictlabel3)

###Q5( part 3,4)  -> Confusion matrix , TNR and TPR for year 2 ie 2020 ####
cf_1 = confusion_matrix( Y_test , predictlabel3 )
print("Confusion matrix for year 2020  for k {} is {} ".format(kvalknn, cf_1))
tpr = cf_1[1][1]/(cf_1[1][1] + cf_1[1][0])
tnr = cf_1[0][0]/(cf_1[0][0] + cf_1[0][1])
print("TPR  for year 2020 is {}  and TNR for year 2020 is {} with KNN".format( tpr, tnr))

print('***** Labels buy and hold and trading Strategy *****')

googd_path = os.path.abspath('GOOG_weekly_return_volatility_detailed.csv')

df_googvold = pd.read_csv(googd_path,parse_dates=["Date"],dayfirst=True).drop(['High','Low','Open','Close','Volume', 'mean_return','volatility'], axis=1).sort_values('Date')
df_googvold['Open'] = df_googvold['Adj Close'].shift(1)
df_googvold['Close'] = df_googvold['Adj Close']
df_googvold = df_googvold.drop(['Adj Close'], axis = 1)

df_googvold = df_googvold[df_googvold.Year.isin([2020])]
df_goog = df_googvold.groupby(['Year','Week_Number']).agg({'Date': ['min','max']}).reset_index()
df_goog.columns = ['Year','Week_Number','OpenDate','CloseDate']
df_goog = (df_goog.merge(df_googvold[['Date','Open']], left_on = 'OpenDate', right_on = 'Date')
      .drop('Date',axis=1)
      .merge(df_googvold[['Date','Close']], left_on = 'CloseDate', right_on = 'Date')
      .drop('Date',axis=1))

df_goog = df_goog.merge(df_googvol_2yrs[['Week_Number','Year']],how='left',left_on=['Week_Number','Year'],right_on = ['Week_Number','Year'])

df_goog['Label'] = predictlabel3
df_goog['NexLabel'] = df_goog['Label'].shift(-1)

cap = 100 + 100*(df_goog.loc[52,'Close'] - df_goog.loc[0,'Open'])/df_goog.loc[0,'Open']
print("GOOG buy-hold  cap for 2020 : {}".format(cap))

cap  = 100
op = 0
for index, row in df_goog.iterrows():
    if row[6] == 1 and op == 0:
        op = row[4]
    if row[6] == 1 and row[7] == 0:
        cap = cap + cap * ((row[5] - op)/op)
        op = 0

strategy_kNN = round(cap)
print("GOOG trading startegy based on label cap for 2020 with KNN: {}".format(cap))



print('####################Q6#####################')
k_lst = [3,5,7,9,11]
error_rate = []  
for val in k_lst :
    desc = ['mean_return', 'volatility','Label']
    X = df_googvol[desc].values
    Y = df_googvol['Label'].values
    X_train ,X_test , Y_train , Y_test = train_test_split (X,Y,test_size =0.5)
    predictlabel3 = []
    Countfinal = 0        
    df_train_2019 = pd.DataFrame(X_train, columns = ['mean_return', 'volatility','Label'])
        
    def X_train_labelpred_hyp(inpdataframe,X_testX, X_testY):
            predictlabel1 = []
            for i in (inpdataframe.index):
                primarydistance =[]
                counter=0
                counter1 = 0
                counter2 =0
                counter3=0
                lst = []
                for j in range(0,len(X_train)):            
                    result = ((inpdataframe.loc[i]['mean_return'])-X_testX)*(X_train[j][0]  - (inpdataframe.loc[i]['mean_return'])) + ((inpdataframe.loc[i]['volatility'])-X_testY)*(X_train[j][1]  - (inpdataframe.loc[i]['volatility']))
                    if (result<0):
                        lst.append(X_train[j][2])                          
                for k in range(0,len(lst)):
                    if(lst[k] == 0):
                        counter = counter+1
                    else:
                        counter1 = counter1 + 1              
                if(counter>counter1):
                    predictlabel1.append(0)
                else:
                    predictlabel1.append(1)                    
            for xyz in range(0,len(predictlabel1)):
                if(predictlabel1[xyz] == 0):
                    counter2 = counter2 + 1
                else:
                    counter3 = counter3 + 1
            if(counter2>counter3):
                return(0)
            else:
                return(1)
    def distancewithTrain_hyperplane(X_testX,X_testY):
            distance = []
            for a in range(0,len(X_train)): 
                pointdistance = ((X_testX-X_train[a][0])**2 + (X_testY-X_train[a][1])**2)*0.5
                distance.append(pointdistance)
            series = pd.Series(distance)
            df_train_2019['distance']= series.values
            seriessort = (df_train_2019.sort_values('distance'))
            seriessorted = seriessort[0:val]
            retval = X_train_labelpred_hyp(seriessorted,X_testX,X_testY)
            return(retval)    
    for i in range(0,len(X_test)):
            distance_label3 = distancewithTrain_hyperplane(X_test[i][0],X_test[i][1])
            predictlabel3.append(distance_label3)
                
    for i in range(0,len(predictlabel3)):
            if(predictlabel3[i]==Y_test[i]):
                Countfinal = Countfinal + 1
    print("The Accuracy is {} when k is {} for 2019 with k Hyperplanes".format(round(Countfinal/len(predictlabel3),2),val) )
    error_rate.append(Countfinal/len(predictlabel3))

print("All Accuracies for 2019 with K Hyperplanes are  : ", error_rate)
plt.figure ( figsize =(10 ,4))
ax = plt. gca ()
ax.xaxis.set_major_locator ( plt.MaxNLocator ( integer = True ))
plt.plot ( range (3 ,13 ,2) , error_rate , color ='red', linestyle ='dotted',
marker ='o', markerfacecolor ='black', markersize =10)
plt.title ('Accuracy vs. k for hyperplanes')
plt.xlabel ('number of neighbors : k')
plt.ylabel ('Accuracy')

max_value = max(error_rate)
max_index = error_rate.index(max_value)
hyperk = k_lst[max_index]
print('Optimal value for k in 2019 is : ', hyperk)

########### Q6 Part 2 - Predicting labels for 2020 using optimal k from 2019 #####################
k_lst = [hyperk]
for val in k_lst :
    desc = ['mean_return', 'volatility','Label']
    X = df_googvol_2yrs[desc].values
    Y = df_googvol_2yrs['Label'].values
    X_train ,X_test , Y_train , Y_test = train_test_split (X,Y,test_size =0.5)
    predictlabel3 = []
    Countfinal = 0        
    df_train_2019 = pd.DataFrame(X_train, columns = ['mean_return', 'volatility','Label'])
        
    def X_train_labelpred_hyp(inpdataframe,X_testX, X_testY):
            predictlabel1 = []
            for i in (inpdataframe.index):
                primarydistance =[]
                counter=0
                counter1 = 0
                counter2 =0
                counter3=0
                lst = []
                for j in range(0,len(X_train)):            
                    result = ((inpdataframe.loc[i]['mean_return'])-X_testX)*(X_train[j][0]  - (inpdataframe.loc[i]['mean_return'])) + ((inpdataframe.loc[i]['volatility'])-X_testY)*(X_train[j][1]  - (inpdataframe.loc[i]['volatility']))
                    if (result<0):
                        lst.append(X_train[j][2])                          
                for k in range(0,len(lst)):
                    if(lst[k] == 0):
                        counter = counter+1
                    else:
                        counter1 = counter1 + 1              
                if(counter>counter1):
                    predictlabel1.append(0)
                else:
                    predictlabel1.append(1)                    
            for xyz in range(0,len(predictlabel1)):
                if(predictlabel1[xyz] == 0):
                    counter2 = counter2 + 1
                else:
                    counter3 = counter3 + 1
            if(counter2>counter3):
                return(0)
            else:
                return(1)
    def distancewithTrain_hyperplane(X_testX,X_testY):
            distance = []
            for a in range(0,len(X_train)): 
                pointdistance = ((X_testX-X_train[a][0])**2 + (X_testY-X_train[a][1])**2)*0.5
                distance.append(pointdistance)
            series = pd.Series(distance)
            df_train_2019['distance']= series.values
            seriessort = (df_train_2019.sort_values('distance'))
            seriessorted = seriessort[0:val]
            retval = X_train_labelpred_hyp(seriessorted,X_testX,X_testY)
            return(retval)    
    for i in range(0,len(X_test)):
            distance_label3 = distancewithTrain_hyperplane(X_test[i][0],X_test[i][1])
            predictlabel3.append(distance_label3)
                
    for i in range(0,len(predictlabel3)):
            if(predictlabel3[i]==Y_test[i]):
                Countfinal = Countfinal + 1
    print("The Accuracy is {} when k is {} for 2020 with k Hyperplanes".format(round(Countfinal/len(predictlabel3),2),val) )
    error_rate.append(Countfinal/len(predictlabel3))
    Acc_Khyper = round(round(Countfinal/len(predictlabel3),2)*100)

print("pred_k_2020", predictlabel3)

###Q6( part 3,4)  -> Confusion matrix , TNR and TPR for year 2 ie 2020 ####
cf_1 = confusion_matrix( Y_test , predictlabel3 )
print("Confusion matrix for year 2020  for k {} is {} ".format(hyperk, cf_1))
tpr = cf_1[1][1]/(cf_1[1][1] + cf_1[1][0])
tnr = cf_1[0][0]/(cf_1[0][0] + cf_1[0][1])
print("TPR  for year 2020 is {}  and TNR for year 2020 is {} with K Hyperplanes".format( tpr, tnr))

print('***** Labels buy and hold and trading Strategy *****')

googd_path = os.path.abspath('GOOG_weekly_return_volatility_detailed.csv')

df_googvold = pd.read_csv(googd_path,parse_dates=["Date"],dayfirst=True).drop(['High','Low','Open','Close','Volume', 'mean_return','volatility'], axis=1).sort_values('Date')
df_googvold['Open'] = df_googvold['Adj Close'].shift(1)
df_googvold['Close'] = df_googvold['Adj Close']
df_googvold = df_googvold.drop(['Adj Close'], axis = 1)

df_googvold = df_googvold[df_googvold.Year.isin([2020])]
df_goog = df_googvold.groupby(['Year','Week_Number']).agg({'Date': ['min','max']}).reset_index()
df_goog.columns = ['Year','Week_Number','OpenDate','CloseDate']
df_goog = (df_goog.merge(df_googvold[['Date','Open']], left_on = 'OpenDate', right_on = 'Date')
      .drop('Date',axis=1)
      .merge(df_googvold[['Date','Close']], left_on = 'CloseDate', right_on = 'Date')
      .drop('Date',axis=1))

df_goog = df_goog.merge(df_googvol_2yrs[['Week_Number','Year']],how='left',left_on=['Week_Number','Year'],right_on = ['Week_Number','Year'])

df_goog['Label'] = predictlabel3
df_goog['NexLabel'] = df_goog['Label'].shift(-1)

cap = 100 + 100*(df_goog.loc[52,'Close'] - df_goog.loc[0,'Open'])/df_goog.loc[0,'Open']
print("GOOG buy-hold  cap for 2020 : {}".format(cap))

cap  = 100
op = 0
for index, row in df_goog.iterrows():
    if row[6] == 1 and op == 0:
        op = row[4]
    if row[6] == 1 and row[7] == 0:
        cap = cap + cap * ((row[5] - op)/op)
        op = 0

strategy_khyper = round(cap)
print("GOOG trading startegy based on label cap for 2020 with K Hyperplanes: {}".format(cap))



print('####################Q7#####################')

df_final = pd.DataFrame({'Method': ['Buy-and-Hold','k-NN (Euclidean, p = 2)','k-NN (Manhattan, p = 1)','k-NN (Minkowski, p = 1:5)', 'Nearest Centroid', 'Domain Transformation', 'k-Predicted Neighbors', 'k-Hyperplanes']*1,
                          'Best k': [0]*8,
                          '% Accuracy': [0]*8,
                          'Amount': [0]*8,
                        })


df_final.loc[df_final["Method"] == 'Buy-and-Hold' ,'Best k'] = 'N/A'
df_final.loc[df_final["Method"] == 'Buy-and-Hold' ,'% Accuracy'] = 'N/A'
df_final.loc[df_final["Method"] == 'Buy-and-Hold' ,'Amount'] = round(cap_bnhold)
df_final.loc[df_final["Method"] == 'k-NN (Euclidean, p = 2)','Best k'] = kvalp2
df_final.loc[df_final["Method"] == 'k-NN (Euclidean, p = 2)','% Accuracy'] = Acc_per_p2 
df_final.loc[df_final["Method"] == 'k-NN (Euclidean, p = 2)','Amount'] = strategy_p2 
df_final.loc[df_final["Method"] == 'k-NN (Manhattan, p = 1)','Best k'] = kvalp1
df_final.loc[df_final["Method"] == 'k-NN (Manhattan, p = 1)','% Accuracy'] = Acc_per_p1
df_final.loc[df_final["Method"] == 'k-NN (Manhattan, p = 1)','Amount'] = strategy_p1
df_final.loc[df_final["Method"] == 'k-NN (Minkowski, p = 1:5)','Best k'] = kvalp105
df_final.loc[df_final["Method"] == 'k-NN (Minkowski, p = 1:5)','% Accuracy'] = Acc_per_p105
df_final.loc[df_final["Method"] == 'k-NN (Minkowski, p = 1:5)','Amount'] = strategy_p105
df_final.loc[df_final["Method"] == 'Nearest Centroid','Best k'] = 'N/A'
df_final.loc[df_final["Method"] == 'Nearest Centroid','% Accuracy'] = Acc
df_final.loc[df_final["Method"] == 'Nearest Centroid','Amount'] = strategy_centroid
df_final.loc[df_final["Method"] == 'Domain Transformation','Best k'] = kvaldt
df_final.loc[df_final["Method"] == 'Domain Transformation','% Accuracy'] = Acc_per_dt
df_final.loc[df_final["Method"] == 'Domain Transformation','Amount'] = strategy_dt
df_final.loc[df_final["Method"] == 'k-Predicted Neighbors','Best k'] = kvalknn
df_final.loc[df_final["Method"] == 'k-Predicted Neighbors','% Accuracy'] = Acc_KNN
df_final.loc[df_final["Method"] == 'k-Predicted Neighbors','Amount'] = strategy_kNN
df_final.loc[df_final["Method"] == 'k-Hyperplanes','Best k'] = hyperk
df_final.loc[df_final["Method"] == 'k-Hyperplanes','% Accuracy'] = Acc_Khyper
df_final.loc[df_final["Method"] == 'k-Hyperplanes','Amount'] = strategy_khyper

print(df_final)
df_final.to_csv('summary.csv',index =False)