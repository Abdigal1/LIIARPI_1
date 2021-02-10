from skimage.io import imsave
from Utilities import *
import os

#di='D:\Documentos\LIIARPI\Anemia\Base_de_Datos\Imagenes_Originales\c1anemia-111.jpg'


#img = io.imread(di)
#ROI1=stickerFilter(img)
#n=di.split('\\')
#name="SF_"+n[-1]
#dird=('\\').join(n[0:-2])
#dird=dird+'\Imagenes_sin_sticker'
#imsave(dird+'\\'+name, ROI1)

di='D:\Documentos\LIIARPI\Anemia\Base_de_Datos\Imagenes_Originales'

imgs=os.listdir(di)
print(imgs)

for imgd in imgs:
    ddi=di+'\\'+imgd
    img = io.imread(ddi)
    ROI1=stickerFilter(img)
    n=di.split('\\')
    name="SF_"+n[-1]
    dird=('\\').join(n[0:-2])
    dird=dird+'\Imagenes_sin_sticker'
    imsave(dird+'\\'+name, ROI1)