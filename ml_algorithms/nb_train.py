#Various imports that will be required for data handling and plotting
import pandas as pd

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
mtp = train["type"]

train = train.drop(["type","id"],axis='columns')
test = test.drop(["id"],axis='columns')

from sklearn.model_selection import train_test_split #Split our data into training and test sets.
from mixed_naive_bayes import MixedNB
from sklearn.metrics import classification_report,confusion_matrix#Report the accuracy scores and the effectiveness of our model.

"""
xtrain contains the training set's features.
ytrain contains the training set's labels.
xtest contains the test set's features.
ytest contains the test set's labels.
"""
#xtrain, xtest, ytrain, ytest = train_test_split(train, mtp, test_size=0.3, random_state=42)
xtrain, xtest, ytrain, ytest = train_test_split(train, mtp, test_size=0.3)

#Drop unnecessary columns
nb = MixedNB(categorical_features=[4])
nb.fit(xtrain,ytrain)
ypred = nb.predict(xtest)

        
print("##########VARIOUS SCORE METRICS:#############\n")
print(classification_report(ytest,ypred))
print("#############################################")
        
print("\n\n#####CONFUSION MATRIX:#####")
print(confusion_matrix(ytest, ypred))#See knn_train for confusion_matrix explanation.
print("###########################")
        
tmp_train_score = nb.score(xtrain,ytrain)
tmp_test_score = nb.score(xtest,ytest)

print("\n\n####TRAIN-ACCURACY-SCORE METRIC####")
print(tmp_train_score)
print("\n\n####TEST-ACCURACY-SCORE-METRIC####")
print(tmp_test_score)
        

#Kaggle Prediction
final_pred = nb.predict(test)

solution = pd.DataFrame()
solution['id'] = test_index
solution['type'] = final_pred
    
#decode
solution['type'] = type_label_encoder.inverse_transform(solution['type'])
solution.to_csv("submission.csv", index=False)

#KAGGLE ACCURACIES#
#Kaggle accuracy using a Naive Bayes model. 
#Probabilities are calculated by the continuous variables 'bone_length, rotting_flesh,hair_length,has_soul' and the categorical value 'color'.
#0.72778