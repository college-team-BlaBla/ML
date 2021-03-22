import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
import random
import math
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def Rand_on_Max_Min (arr, idx):
    maxx = max(arr[:, idx])
    minn = min(arr[:, idx])
    random.seed(0)
    #print(maxx,minn)
    for i in arr:
        if math.isnan(i[idx]):
            i[idx] = random.randint(minn,maxx)
            #print(i[idx])

    pass

def UnknownCode (arr, idx):
    maxx = max(arr[:, idx])
    for i in arr:
        if math.isnan(i[idx]):
            i[idx] = maxx+1

    pass



def PreProcessing () :
    data = pd.read_csv('Data.csv')
    data.drop(["id", "name", "release_date"], axis=1, inplace = True)

    #print(data)
    X = data.iloc[:, :-1].values
    Y = data.iloc[:, -1].values



    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    imputer.fit(X[:, 0:1])

    X[:, 0:1] = imputer.transform(X[:,0:1])

    ###########

    imputer.fit(X[:, 2:3])
    X[:, 2:3] = imputer.transform(X[:, 2:3])

    ###########

    imputer.fit(X[:, 4:7])
    X[:, 4:7] = imputer.transform(X[:, 4:7])

    ###########
    imputer.fit(X[:, 8:12])
    X[:, 8:12] = imputer.transform(X[:, 8:12])

    ###########
    imputer.fit(X[:, 13:15])
    X[:, 13:15] = imputer.transform(X[:, 13:15])

    labelencoder = LabelEncoder()
    X[:, 3] = labelencoder.fit_transform(X[:, 3])

    Rand_on_Max_Min(X, 1)
    UnknownCode(X, 3)
    Rand_on_Max_Min(X, 7)
    Rand_on_Max_Min(X, 12)
    Y = labelencoder.fit_transform(Y)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1)

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)




    Out = pd.DataFrame(X_train)
    Out.to_csv("X_train.csv")

    Out = pd.DataFrame(X_test)
    Out.to_csv("X_test.csv")

    return  X_train, X_test, Y_train, Y_test

