import os
import numpy as np
import pandas as pd
import argparse
import json
from sklearn.model_selection import cross_validate
from sklearn.svm import SVR
from sklearn.linear_model import Lasso
from sklearn.linear_model import BayesianRidge
import matplotlib.pyplot as plt
from sklearn.svm import SVC
plt.rcParams["figure.figsize"] = (20,3)
def parse_args():
    parser = argparse.ArgumentParser('Train')
    parser.add_argument('--dataset', type=str, default=os.path.join(os.path.curdir, 'Metadata_V7G_sckit'), help='Path of Dataset')
    parser.add_argument('--parameters', type=str, default='def.json', help='Trainning parameters', )
    parser.add_argument('--normalized', type=bool, default=False, help='Specifies if data is normalized')
    return parser.parse_args()


def load(dat, args):
    N = 5
    PATH = args.dataset
    bT = 0
    #key_met = [line.strip() for line in open("meta.txt")]
    df = pd.read_csv("db_sample_201901221525.csv")
    Y = []
    X = []
    for file in os.listdir(PATH):
        lo = file.split("_")[1][:-4] + ".jpg"
        val = df.loc[df["imagename"]==lo, "ane_glo"].iloc[0]
        dictdata  = np.load(os.path.join(PATH,file), allow_pickle=True)[()]
        if len(dictdata)<N:
            #print("N menor a %d"% N)
            bT += 1
            continue
        if val>20 or val<5:
            continue
        Y.append(val)
        #print(file, val, type(dictdata))
        aux = []
        #print(type(dictdata))
        for i in dictdata.keys():
            nn = dictdata[i]
            for j in dat:
                if '.' in j:
                    key_m = j.split(".")[0]
                    key_n = int(j.split(".")[-1])
                    aux.append(nn[key_m][key_n])
                else:
                    try:
                        aux.extend(nn[j])
                    except:
                        aux.append(nn[j])
            


        X.append(aux)
    X = np.array(X)
    y = np.array(Y)
    print(f"Total de datos menor a 5 : {bT} de {len(X)}")

    return X, y

def main(args):
    with open(args.parameters) as f:
        data = json.load(f)

    trains = data['train']
    cc = 1
    for di in trains.values():
        X, y = load(di["metadata"], args)
        ND = X.shape[0]
        m_s = di["model"]
        model = None
        if m_s == "SVR":
            model = SVR(C=1.0, kernel="rbf")
        elif m_s == "Lasso":
            model = Lasso(max_iter=3000, alpha=0.2)
        elif m_s == "SVC":
            
            model = SVC(kernel = 'rbf')
            y = y>10
            y = y.astype(int)
            plt.hist(y)
            plt.show()
        else:
            model = BayesianRidge(lambda_1=0.01, lambda_2=1e-6)
        #print(X.shape, y.shape)
        out = cross_validate(model, X, y, scoring=tuple(di["metrics"]), cv=int(di["nfolds"]), return_estimator=True)
        print(out.keys())
        print("==========================SCORE=====================")
        for i in range(3, len(out.keys())):
            scores = np.array(out[list(out.keys())[i]])
            print(list(out.keys())[i])
            print(scores)
        score_bound = scores.mean()
        ax = plt.figure(cc)
        mods = out['estimator'][-1]
        yhat = mods.predict(X)
        ts = np.random.randint(0, ND, (200,))
        x = np.arange(200)
        plt.plot(x, y[ts])
        plt.plot(x, yhat[ts])
        #plt.plot(x, yhat[ts] + score_bound)
        #plt.plot(x, yhat[ts] - score_bound)
        plt.fill_between(x, yhat[ts] - score_bound, yhat[ts] + score_bound, facecolor='yellow', alpha=0.5)
        #plt.legend(["Real", "Estimado", "Superior", "Inferior"])
        plt.legend(["Real", "Estimado"])
        plt.ylim([8, 16])
        ax.show()
        cc+=1
    input()

if __name__ == "__main__":
    
    args = parse_args()
   
    main(args)




