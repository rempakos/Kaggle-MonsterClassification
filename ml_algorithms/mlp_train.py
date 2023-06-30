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
from sklearn.metrics import classification_report,confusion_matrix#Report the accuracy scores and the effectiveness of our model.


"""
xtrain contains the training set's features.
ytrain contains the training set's labels.
xtest contains the test set's features.
ytest contains the test set's labels.
"""

#xtrain, xtest, ytrain, ytest = train_test_split(train, mtp, test_size=0.3) 
xtrain, xtest, ytrain, ytest = train_test_split(train, mtp, test_size=0.3, random_state=42)

import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras import layers, Model
from tensorflow.keras import activations
from tensorflow.keras.utils import to_categorical

# Create a `Sequential` model. If isAlpha is 'True' then we have 1 hidden layer, else 2 hidden layers.
def create_model(isAlpha,activ_func,K1,K2):

    if(isAlpha == True):
        model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(5,)),
            layers.Dense(units = K1,activation=activ_func),
            layers.Dense(units = 3,activation='softmax')
            ])
        return model
    elif(isAlpha == False):
        model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(5,)),
            layers.Dense(units = K1,activation=activ_func),
            layers.Dense(units = K2,activation=activ_func),
            layers.Dense(units = 3,activation='softmax')
            ])
        return model
    else:
        print("Invalid model structure given, please run with 'True' for 'alpha' block or 'False' for 'beta' block.")


def mlp_classify(xtrain,xtest,ytrain,ytest):
    #PARAMETERS FOR OUR MODEL FRAMEWORK
    parameters = {
        "batch_size": 1, #keep at 1 for SGD
        "num_epoch": 100,
        "activation_func": 'sigmoid', # 'sigmoid' or 'tanh'
        "K1": 200, #Number of neurons per layer
        "K2": 100, #Number of neurons per layer
        }
    activ_func = parameters["activation_func"]
    K1 = parameters["K1"]
    K2 = parameters["K2"]
    batch_size = parameters["batch_size"]
    num_epoch = parameters["num_epoch"]
    #
    
    #Same as sklearns label_encoder, we convert ytrain and ytest to categorical values in order for them to have
    #the correct shape.(just using keras to_categorical instead of label_encoder for academic reasons)
    ytrain = to_categorical(ytrain, 3)
    ytest = to_categorical(ytest, 3)
    
    ###########CREATE YOUR MODEL HERE###########
    #False for beta block, True for alphablock
    model = create_model(True,activ_func,K1,K2)
    ############################################
    
    model.compile(optimizer="sgd", loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(xtrain,ytrain,batch_size=batch_size,epochs=num_epoch)
    print("###########################")
    score = model.evaluate(xtest,ytest,verbose=0)#verbose  = 0 since we dont need a progress bar here
    
    print("\n\n#####MODEL ACCURACY:#####")
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    print("###########################")
    
    ypred = model.predict(xtest)
    ypred=np.argmax(ypred, axis=1)

    ytest=np.argmax(ytest, axis=1)
    
    print("\n\n#####CONFUSION MATRIX:#####")
    print(confusion_matrix(ytest, ypred))#See knn_train for confusion_matrix explanation.
    print("###########################")
    
    print("##########VARIOUS SCORE METRICS:#############\n")
    print(classification_report(ytest,ypred))
    print("#############################################")
    
    #Kaggle Prediction
    final_pred = model.predict(test)

    final_pred = np.argmax(final_pred, axis=1)

    solution = pd.DataFrame()
    solution['id'] = test_index
    solution['type'] = final_pred
    
    #decode
    solution['type'] = type_label_encoder.inverse_transform(solution['type'])
    solution.to_csv("submission.csv", index=False)

mlp_classify(xtrain, xtest, ytrain, ytest)

#KAGGLE ACCURACIES#

##########MODEL A##########
#Sigmoid act func#
#Kaggle accuracy for model 'a' of 1 hidden layer of 50 neurons for 100 epochs with sigmoid act. func
#0.68809

#Kaggle accuracy for model 'a' of 1 hidden layer of 100 neurons for 100 epochs with sigmoid act. func
#0.72778

#Kaggle accuracy for model 'a' of 1 hidden layer of 200 neurons for 100 epochs with sigmoid act. func
#0.69754

##########
#Tanh act func#
#Kaggle accuracy for model 'a' of 1 hidden layer of 50 neurons for 100 epochs with tanh act. func
#0.72022

#Kaggle accuracy for model 'a' of 1 hidden layer of 100 neurons for 100 epochs with tanh act. func
#0.70510

#Kaggle accuracy for model 'a' of 1 hidden layer of 200 neurons for 100 epochs with tanh act. func
#0.63705

##########MODEL B##########
#Sigmoid act func#
#Kaggle accuracy for model 'b' of 2 hidden layers of 50 & 25 neurons for 100 epochs with sigmoid act. func
#0.69565

#Kaggle accuracy for model 'b' of 2 hidden layers of 100 & 50 neurons for 100 epochs with sigmoid act. func
#0.70888

#Kaggle accuracy for model 'b' of 2 hidden layers of 200 & 100 neurons for 100 epochs with sigmoid act. func
#0.69943

##########
#Tanh act func#

#Kaggle accuracy for model 'b' of 2 hidden layers of 50 & 25 neurons for 100 epochs with tanh act. func
#0.72967

#Kaggle accuracy for model 'b' of 2 hidden layers of 100 & 50 neurons for 100 epochs with tanh act. func
#0.73534

#Kaggle accuracy for model 'b' of 2 hidden layers of 200 & 100 neurons for 100 epochs with tanh act. func
#0.69376