# Data Preprocessing
import numpy as np
import pandas as pd

df = pd.read_csv('cdata_new.csv')
print ("original df.head() ", df.shape)

df.head()

#Encode binary categorical vars

gender_dict = {
    'MALE': 1,
    'FEMALE':0
}

new_cus_dict = {
    '1. Yes': 1,
    '2. No': 0
}

pre_facility_dict = {
    '1. Yes': 1,
    '2. No': 0
}

pre_guaranter_dict = {
    '1 Yes': 1,
    '2 No': 0
}

status_dic = {
    'Regular': 1,
    'NPA': 0
}

df['GENDER'] = df['GENDER'].map(gender_dict)
df['IS_NEW_CUS'] = df['IS_NEW_CUS'].map(new_cus_dict)
df['PREVIOUS_FACILITY'] = df['PREVIOUS_FACILITY'].map(pre_facility_dict)
df['PREVIOUS_GUARANTER'] = df['PREVIOUS_GUARANTER'].map(pre_guaranter_dict)
df['STATUS'] = df['STATUS'].map(status_dic)

df.head()

cus_dic = {
    'Normal Customer': 'NC',
    'Fully Secured':'FS',
    'Staff': 'STAFF'
}

df['CUS_CAT'] = df['CUS_CAT'].map(cus_dic)

df.columns

df = pd.get_dummies(df,prefix=['CUS_CAT'], columns = ['CUS_CAT'])
df = pd.get_dummies(df,prefix=['EMP_TYPE'], columns = ['EMP_TYPE'])
df = pd.get_dummies(df,prefix=['NUM_CC_HELD'], columns = ['NUM_CC_HELD'])
df = pd.get_dummies(df,prefix=['YEARS_AT_ADDRESS'], columns = ['YEARS_AT_ADDRESS'])
df = pd.get_dummies(df,prefix=['REL_WITH_BANK'], columns = ['REL_WITH_BANK'])
df = pd.get_dummies(df,prefix=['ACCOMADATION_TYPE'], columns = ['ACCOMADATION_TYPE'])
df = pd.get_dummies(df,prefix=['YEARS_IN_JOB'], columns = ['YEARS_IN_JOB'])
df = pd.get_dummies(df,prefix=['CRIB_BORROWER'], columns = ['CRIB_BORROWER'])
df = pd.get_dummies(df,prefix=['CRIB_GUARANTOR'], columns = ['CRIB_GUARANTOR'])
df = pd.get_dummies(df,prefix=['AGE_CATEGORY'], columns = ['AGE_CATEGORY'])

df['CUR_AGE'] = 2021 -  df['BIRTHYEAR']
df.drop('BIRTHYEAR', inplace=True, axis=1)

# apply normalization techniques

from sklearn.preprocessing import MinMaxScaler

column = 'INCOME'
df[column] = MinMaxScaler().fit_transform(np.array(df[column]).reshape(-1,1))

column = 'EXISTING_LOAN_INSTALLMENTS'
df[column] = MinMaxScaler().fit_transform(np.array(df[column]).reshape(-1,1))

df.to_csv('data_2021_10_31.csv',index=False)

import pandas as pd
df = pd.read_csv ('data_2021_10_31.csv')
x = df.drop (["STATUS"], axis=1).values
y = df ["STATUS"].values

# Final Model

import pandas as pd
import numpy as np
import pickle
import sys
import warnings

import numpy as np
import pandas as pd

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn import linear_model

from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif

#use cross validation and feature selection

from sklearn.model_selection import cross_val_score

def Anova_Feature_selction (df,Ratio,No_of_Variable):
    X,y =  cc.loc[:, cc.columns != 'STATUS'], cc.STATUS     
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=Ratio, random_state=0)

    bestfeatures = SelectKBest(score_func=f_classif, k=No_of_Variable)
    fit = bestfeatures.fit(X_train,y_train)
    dfscores = pd.DataFrame(fit.scores_)
    dfcolumns = pd.DataFrame(X_train.columns)
    #concat two dataframes for better visualization 
    featureScores = pd.concat([dfcolumns,dfscores],axis=1)
    featureScores.columns = ['Specs','Score']  #naming the dataframe columns
    return (featureScores.nlargest(No_of_Variable,'Score'))  #print 10 best features

def Chi_Feature_selction (df,Ratio,No_of_Variable):
    X,y =  cc.loc[:, cc.columns != 'STATUS'], cc.STATUS     
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=Ratio, random_state=0)

    bestfeatures = SelectKBest(score_func=chi2, k=No_of_Variable)
    fit = bestfeatures.fit(X_train,y_train)
    dfscores = pd.DataFrame(fit.scores_)
    dfcolumns = pd.DataFrame(X_train.columns)
    #concat two dataframes for better visualization 
    featureScores = pd.concat([dfcolumns,dfscores],axis=1)
    featureScores.columns = ['Specs','Score']  #naming the dataframe columns
    df=featureScores.nlargest(No_of_Variable,'Score')
    df=df[df.Specs != "STATUS"]
    return (df)  #print 10 best features
    
if not sys.warnoptions:
    warnings.simplefilter("ignore")
    
cc = pd.read_csv('data_2021_10_31.csv')
cc.drop(['CUR_AGE'], axis = 1)

# Feature Selction 

Cato_Data_set = cc.drop(['INCOME'], axis = 1)
Cato_Data_set = Cato_Data_set.drop(['EXISTING_LOAN_INSTALLMENTS'], axis = 1)

print ("\n\n=============================================")
print ("Cato_Data_set.shape",Cato_Data_set.shape)
Cato_Data_set_after = Chi_Feature_selction (Cato_Data_set,0.2,10) ## parameter 10
print ("Cato_Data_set_after list",Cato_Data_set_after)

for i in range(len(Cato_Data_set_after["Specs"])):
    A_Specs =Cato_Data_set_after["Specs"]
    B_Score =Cato_Data_set_after["Score"]
    

Best_Cato_names = Cato_Data_set_after["Specs"].tolist()

Best_Cato_names = Best_Cato_names + ["INCOME","EXISTING_LOAN_INSTALLMENTS","STATUS"]

cc=cc[Best_Cato_names]
print("bb.shape", cc.shape)

# Import train_test_split
from sklearn.model_selection import train_test_split

# Segregate features and labels into separate variables
X,y =  cc.loc[:, cc.columns != 'STATUS'], cc.STATUS 

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X,
                                y,
                                test_size=0.2,
                                random_state=0)

print("X_train.shape", X_train.shape)
print("y_train.shape", y_train.shape)

# Import confusion_matrix
from sklearn.metrics import confusion_matrix

from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)
RandomForest_Cross_Accuracy= (cross_val_score(rf, X_train, y_train, cv=5))
print("RandomForest_Cross_Accuracy :  ",RandomForest_Cross_Accuracy)

pickle.dump(rf, open('model.pkl','wb'))

nb = GaussianNB()
nb.fit(X_train, y_train)
GaussianNB_Cross_Accuracy= (cross_val_score(nb, X_train, y_train, cv=5))
print("GaussianNB_Cross_Accuracy :  ",GaussianNB_Cross_Accuracy)

kn = KNeighborsClassifier(n_neighbors=3)
kn.fit(X_train, y_train)
KNeighbors_Cross_Accuracy= (cross_val_score(kn, X_train, y_train, cv=5))
print("KNeighbors_Cross_Accuracy :  ",KNeighbors_Cross_Accuracy)

### Prediction probabilities 

r_probs = [0 for _ in range(len(y_test))]
rf_probs = rf.predict_proba(X_test)
nb_probs = nb.predict_proba(X_test)
kn_probs = kn.predict_proba(X_test)

rf_probs = rf_probs[:, 1]
nb_probs = nb_probs[:, 1]
kn_probs = kn_probs[:, 1]

from sklearn.metrics import roc_curve, roc_auc_score

r_auc = roc_auc_score(y_test, r_probs)
rf_auc = roc_auc_score(y_test, rf_probs)
nb_auc = roc_auc_score(y_test, nb_probs)
kn_auc = roc_auc_score(y_test, kn_probs)

print('Random (chance) Prediction: AUROC = %.3f' % (r_auc))
print('Random forest: AUROC = %.3f' % (rf_auc))
print('Naive Bayes: AUROC = %.3f' % (nb_auc))
print('KNeighbors : AUROC = %.3f' % (kn_auc))

r_fpr, r_tpr, _ = roc_curve(y_test, r_probs)
rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_probs)
nb_fpr, nb_tpr, _ = roc_curve(y_test, nb_probs)
kn_fpr, kn_tpr, _ = roc_curve(y_test, kn_probs)

import matplotlib.pyplot as plt

plt.plot(r_fpr, r_tpr, linestyle='--', label='Random prediction (AUROC = %0.3f)' % r_auc)
plt.plot(rf_fpr, rf_tpr, marker='.', label='Random Forest (AUROC = %0.3f)' % rf_auc)
plt.plot(nb_fpr, nb_tpr, marker='.', label='Naive Bayes (AUROC = %0.3f)' % nb_auc)
plt.plot(kn_fpr, kn_tpr, marker='.', label='KNeighbors (AUROC = %0.3f)' % kn_auc)

# Title
plt.title('ROC Plot')
# Axis labels
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
# Show legend
plt.legend() # 
# Show plot
plt.show()
