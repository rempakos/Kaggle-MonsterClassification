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

#Various imports that will be required to build our model framework.
from sklearn.metrics import classification_report,confusion_matrix#Report the accuracy scores and the effectiveness of our model.
from sklearn.model_selection import train_test_split #Split our data into training and test sets.
from sklearn.neighbors import KNeighborsClassifier #KNeighbors method.
from sklearn.linear_model import LogisticRegression #LR

"""
xtrain contains the training set's features.
ytrain contains the training set's labels.
xtest contains the test set's features.
ytest contains the test set's labels.
"""
###Split our dataset into a training and test set.Also shuffle our data.
xtrain, xtest, ytrain, ytest = train_test_split(train, mtp, test_size=0.3, random_state=42)

#Classification report of our various accuracies.
#KNN.score(accuracy of correct predictions/total predictions) is not always a good enough metric.
#This can be the case in a biased dataset. 
#Let us consider a dataset of 100 monsters that contained 99 records of ghouls and one record of a ghost.
#Suppose that we have created a false model that somehow 'ALWAYS' predicts a Ghoul monster type.
#Then by using the above testing set our model should have 99% percent accuracy!
#That is obviously wrong and this is why we use various metrics


"""
THIS IS THE CONFUSION MATRIX GIVEN BELOW
IT MEASURES THE EFFECTIVENESS OF OUR MODEL
EXAMPLE
           ---Predicted Values---
-A        |Ghost|    |Ghoul|     |Goblin|
-c|Ghost|   26          0           9
-t
-u
-a
-l
- |Ghoul|    0         27           14
-V
-a
-l
-u
-e
-s|Goblin|   3          6           27
"""
print("\n\n#####CONFUSION MATRIX:#####")
#print(confusion_matrix(ytest, ypred))
print("###########################")

#Assign the number of neighbours as per the given instructions.
#Create 2 numpy arrays that will hold the scores for each different K-neighbor size.
neighbors = np.array([1,3,5,10])
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))


#Perform K-NN classification for neighbor counts.
for i, k in enumerate(neighbors):
    knn = KNeighborsClassifier(n_neighbors = k)

    knn.fit(xtrain, ytrain) #Fit and predict on training/testing sets.
    train_accuracy[i] = knn.score(xtrain, ytrain)
    test_accuracy[i] = knn.score(xtest, ytest)
    ypred = knn.predict(xtest)
    print("##########VARIOUS SCORE METRICS:#############\n")
    print(classification_report(ytest,ypred))
    print("#############################################")
    
    print("\n\n#####CONFUSION MATRIX:#####")
    print(confusion_matrix(ytest, ypred))
    print("###########################")

    
print("\n\n####TRAIN-ACCURACY-SCORE METRIC FOR VARIOUS NEIGHBOURS####")
print(train_accuracy)

print("\n\n####TEST-ACCURACY-SCORE-METRIC FOR VARIOUS NEIGHBOURS####")
print(test_accuracy)


def perform_knn_classification(kneighbours):
    knn = KNeighborsClassifier(n_neighbors = kneighbours)
    knn.fit(xtrain, ytrain) #Fit and predict on training/testing sets.
    
    #Kaggle Prediction
    final_pred = knn.predict(test)

    solution = pd.DataFrame()
    solution['id'] = test_index
    solution['type'] = final_pred
            
    #decode
    solution['type'] = type_label_encoder.inverse_transform(solution['type'])

    solution.to_csv("submission.csv", index=False)

perform_knn_classification(10)

#KAGGLE ACCURACIES#
#Kaggle accuracy using a K-Nearest-Neighbour model with k=1.
#0.67674

#Kaggle accuracy using a K-Nearest-Neighbour model with k=3.
#0.67296

#Kaggle accuracy using a K-Nearest-Neighbour model with k=5.
#0.68998

#Kaggle accuracy using a K-Nearest-Neighbour model with k=10.
#0.68620
