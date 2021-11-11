import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import os

PATH = os.path.join(os.path.curdir, '..\\Data_base\\Metadata_V9G_sckit')
flaux = True
PARAM = None
N = 1
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
        if flaux:
                PARAM = list(nn.values()) 
                flaux = False
        for i in nn.values():
            
            try:
                aux.extend(i)
            except:
                aux.append(i)
        #break


    X.append(aux)
X = np.array(X)
y = np.array(Y)
print(f"Total de datos menor a 5 : {bT} de {len(X)}")
print(X.shape, y.shape)
for i in range(X.shape[1]):
    corr, _ = pearsonr(X[:, i], y)
    #if abs(corr) >0.2:
    print(i, corr)
    