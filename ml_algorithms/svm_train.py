#Various imports that will be required for data handling and plotting
import pandas as pd
import numpy as np

##Read our training set and Test set, assign the 'heads' of those files(Dataframes).
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
train.head()
test.head()


#Encode our color & type columns via sklearns' label_encoder.
from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()
type_label_encoder = preprocessing.LabelEncoder()
train['color'] = label_encoder.fit_transform(train['color'])
train['type'] = type_label_encoder.fit_transform(train['type'])


test['color'] = label_encoder.fit_transform(test['color'])
test_index = test['id']

mtp = train["type"]#Keep our labels, they'll be handy later!

#Drop unnecessary columns
train = train.drop(["type","id"],axis='columns')
test = test.drop(["id"],axis='columns')

from sklearn.model_selection import train_test_split #Split our data into training and test sets.
from sklearn.svm import SVC #Use Support Vector Machine to find our hyperplanes
from sklearn.metrics import classification_report,confusion_matrix#Report the accuracy scores and the effectiveness of our model.

"""
xtrain contains the training set's features.
ytrain contains the training set's labels.
xtest contains the test set's features.
ytest contains the test set's labels.
"""
#xtrain, xtest, ytrain, ytest = train_test_split(train, mtp, test_size=0.3, random_state=42)
xtrain, xtest, ytrain, ytest = train_test_split(train, mtp, test_size=0.3)

#Support Vector Machine model(Find the ndimension hyperplanes to classify our monster types).
#if kernel = True use 'rbf' else 'linear'
#decision_function_shape='ovr' | for one versus rest(all) approach
def svm_classification(xtrain,ytrain,xtest,ytest,kernel):
    if(kernel==True):
        svm_model = SVC(C=1400,decision_function_shape='ovr')#default is already rbf, no need to declare it.
        svm_model.fit(xtrain,ytrain)
        
        ypred = svm_model.predict(xtest)
        svm_report(svm_model,ytest,ypred,xtrain,ytrain)
    elif(kernel==False):
        svm_model = SVC(C=1400, kernel = "linear",decision_function_shape='ovr')#Using linear kernel
        svm_model.fit(xtrain,ytrain)
        
        ypred = svm_model.predict(xtest)
        svm_report(svm_model,ytest,ypred,xtrain,ytrain)

        
def svm_report(model,ytest,ypred,xtrain,ytrain):
        print("##########VARIOUS SCORE METRICS:#############\n")
        print(classification_report(ytest,ypred))
        print("#############################################")
        
        print("\n\n#####CONFUSION MATRIX:#####")
        print(confusion_matrix(ytest, ypred))#See knn_train for confusion_matrix explanation.
        print("###########################")
        
        tmp_train_score = model.score(xtrain,ytrain)
        tmp_test_score = model.score(xtest,ytest)
        print("\n\n####TRAIN-ACCURACY-SCORE METRIC####")
        print(tmp_train_score)
        print("\n\n####TEST-ACCURACY-SCORE-METRIC####")
        print(tmp_test_score)
        
        #Kaggle Prediction
        final_pred = model.predict(test)

        solution = pd.DataFrame()
        solution['id'] = test_index
        solution['type'] = final_pred
            
        #decode
        solution['type'] = type_label_encoder.inverse_transform(solution['type'])
        solution.to_csv("submission.csv", index=False)

def svm_prepare(functionBool):
    if(functionBool==True):
        print("\n\n##########Radiant Basis Function:#############\n")
        svm_classification(xtrain, ytrain, xtest, ytest, True) #SVM with radiant based function
    else:
        print("\n\n##########Linear Function:#############\n")
        svm_classification(xtrain, ytrain, xtest, ytest, False) #SVM with linear based function
        
#SELECT LINEAR OR RADIANT BASIS KERNEL FUNCTION HERE(LN = False, RB = True)
svm_prepare(False)

#KAGGLE ACCURACIES#
#Kaggle accuracy using a Support Vector Machine model with a Linear Kernel Function.
#0.73534

#Kaggle accuracy using a Support Vector Machine model with a Radiant Basis Kernel Function.
#0.71644