import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor


def load_meta(N = 17):
    df = pd.read_csv("db_sample_201901221525.csv")
    Y = []
    X = []
    for file in os.listdir("Metadata"):
        lo = file.split(".")[0] + ".jpg"
        val = df.loc[df["imagename"]==lo, "ane_glo"].iloc[0]
        dictdata  = np.load(os.path.join(os.getcwd(), "Metadata/"+file), allow_pickle=True)[()]
        Y.append(val)
        #print(file, val, type(dictdata))
        aux = []
        for i in range(N):
            nn = dictdata[i+1]
            #aux.extend(nn["Per"].reshape(-1,).tolist())
            aux.extend(nn["X"].reshape(-1,).tolist())
            aux.extend(nn["Per"].reshape(-1,).tolist())
            aux.extend(nn["Mo"].reshape(-1,).tolist())
            aux.extend(nn["U"].reshape(-1,).tolist())
        
        X.append(aux)
    return np.array(X), np.array(Y)


if __name__ == '__main__':
    

    X, y = load_meta()
    print(X.shape, y.shape)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
    regressor = RandomForestRegressor(max_depth=10, random_state=0)
    #regressor = SVR(kernel = 'poly', degree = 6)
    regressor.fit(X_train, y_train)

    y_pred = regressor.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    print(r2)