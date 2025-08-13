import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report
import pickle  #built-in and works fine for Python 3.8+
def lr(model):
    m1 = -(model.coef_[0][0]/model.coef_[0][1])
    b1 = -(model.intercept_[0][0]/model.coef_[0][1])
    x_inp = np.linspace(-3,3,100)
    y_inp = m1*x_inp + b1
def sigmoid(z):
    return 1/(1 + np.exp(-z))
def gd(X,y):
    X = np.insert(X,0,1,axis=1)
    weights = np.ones(X.shape[1])
    lr = 0.5
    for i in range(2500):
        y_hat = sigmoid(np.dot(X, weights))#y_hat = predicted y
        weights = weights + lr*(np.dot((y - y_hat), X)/X.shape[0])
    return weights[1:], weights[0]

def create_model(data):
    X = data.drop(['diagnosis'], axis = 1)
    Y = data['diagnosis']
    #SCALE THE DATA

    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    #SPLIT THE DATA
    X_train,X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    #TRAIN
    model = LogisticRegression()
    # lr(model)
    # gd(X,Y)
    model.fit(X_train, Y_train)

    #TEST THE MODEL
    y_pred = model.predict(X_test)
    print(f'Accuracy of the model : {accuracy_score(Y_test, y_pred) * 100:.2f} %')
    print('Classification report : ', classification_report(Y_test, y_pred))

    return model, scaler


def get_clean_data():
    data = pd.read_csv('data/data.csv')
    data = data.drop(["Unnamed: 32", "id" ], axis =1)

    data.diagnosis = [1 if value == "M" else 0 for value in data.diagnosis]

    print(data.head())
    return data

def main():

    data = get_clean_data()
    model, scaler = create_model(data)

    with open('model/model.pkl', 'wb') as f: #f means file, w means write, b means binary file
        pickle.dump(model,f)
    with open('model/scaler.pkl', 'wb') as f:
        pickle.dump(scaler,f)

if __name__ == "__main__":
    main()