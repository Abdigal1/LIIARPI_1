import sys
sys.path.append('D:\\Documentos\\LIIARPI\\Anemia\\GH\\LIIARPI_1\\Utils')
from Utilities import *
import numpy as np
from skimage import io
from skimage import color
import matplotlib.pyplot as plt
from skimage.segmentation import slic
from skimage import morphology
import json

f1=open("D:\\Documentos\\LIIARPI\\Anemia\\Base_de_Datos\\validcrop.txt","r")
lines=f1.readlines()
linesn=np.array(lines)
linesn=np.delete(lines,np.where(linesn=="\n"))
linesn=linesn.reshape(-1,3)
linesnc=v_replace_err(linesn)
linesnc=np.vectorize(pyfunc=lambda x:np.array([x[0].split('\n')[0]]),signature="(n)->(m)")(linesnc.reshape(-1,1)).reshape(-1,3)
xywh=linesnc[:,:2]
imgnames=linesnc[:,2]
xywh=v_no_spaces(xywh)
xywh=np.vectorize(pyfunc=(lambda x:float(x)))(xywh.reshape(1,-1)[0])
xywh=xywh.reshape(-1,4).astype(int)+1

dir_origin='D:\\Documentos\\LIIARPI\\Anemia\\Base_de_Datos\\Imagenes_Originales\\'
dir_ROI='D:\\Documentos\\LIIARPI\\Anemia\\Base_de_Datos\\Sem_Auto\\eye_'
dir_meta='D:\\Documentos\\LIIARPI\\Anemia\\Base_de_Datos\\Metadata\\'
#FOR
for name in imgnames:
    try:
        img = io.imread(dir_origin+name)
        ROI = io.imread(dir_ROI+name)
        mask=assemble_mask(xywh[np.where(imgnames==(name))][0],img,ROI)
        lum = np.mean(mask,axis=2).astype(int)
        mask1=lum > 0
        SD=get_Statistical_Descriptors(img,mask1,n_segments=20)
        np.save(dir_meta+name.split('.')[0]+'.npy',SD)
    except:
        print(name)