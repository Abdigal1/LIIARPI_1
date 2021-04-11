import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import axes
from PIL import Image
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage import io
import matplotlib.pyplot as plt
import argparse
from skimage.color import rgb2hsv
from skimage.morphology import convex_hull_image
from sklearn.decomposition import PCA
from skimage import measure
from skimage import color

def maxmod(p,th=5000):
  for i in range(len(p[0][:])):
    if p[0][-i]>th:
      max=p[1][-i]
      break
  return max

def cmask(mask):
  sel=0
  for i in mask.ravel():
    if i:
      sel=sel+1
  return sel

def ucut(img,f):
  cx=int(img.shape[0]/(f*2))
  cy=int(img.shape[1]/(f*2))
  return img[cx:(img.shape[0]-cx),cy:(img.shape[1]-cy)]

def Assem(img,cimg,f):
  m=np.full(img.shape, False)
  cx=int(img.shape[0]/(f*2))
  cy=int(img.shape[1]/(f*2))
  m[cx:img.shape[0]-cx,cy:img.shape[1]-cy]=cimg
  return m

def get_Val(img):
  return img[:, :, 2]

def prePro(img):
  hsv_i= rgb2hsv(img)
  hsv_i = cv2.blur(hsv_i,(25,25),0)
  return get_Val(hsv_i)

def get_binary(img,th=0.8):
  return img > th

def getBiggestCont(binaryImg,n=3):
  contours = measure.find_contours(binaryImg, 0.8)
  contours.sort(key=lambda x: x.shape[0], reverse=True)
  return contours[0:n]

def is_In(In,Ou):
  if ((In[0,0]>Ou[:,0].min())and(In[0,0]<Ou[:,0].max())and(In[0,1]>Ou[:,1].min())and(In[0,1]<Ou[:,1].max())):
    return True
  else:
    return False

def is_Con(Contour,img):
  if((Contour[:,0].min()==0)or(Contour[:,1].min()==0)or(Contour[:,0].max()==(img.shape[0]-1))or(Contour[:,0].max()==(img.shape[1]-1))):
    return False
  else:
    return True

def closedCon(contours,binary_img):
  temp=[]
  for c in contours:
    if is_Con(c,binary_img):
      temp.append(c)
  return temp

def contenedContours(contours):
  cont=[]
  for ic in contours:
    for cc in contours:
      if np.array_equal(ic,cc):
        break
      if is_In(ic,cc):
        cont.append(ic)
        break
  return cont

def nonContened(contours,contened):
  conte=contours
  for c in contened:
    conte.remove(c)
  return conte

def containerContours(Noncontent,Content):
  temp=[]
  for cc in Noncontent:
    for ic in Content:
      if is_In(ic,cc):
        temp.append(cc)
        break
  return temp

def similarForms(contours,pcaModel):
  Meta=[]
  for c in contours:
    PC = pcaModel.fit_transform(c)
    Cc = np.expand_dims(PC.astype(np.float32), 1)
    Cc = cv2.UMat(Cc)
    areaR = cv2.contourArea(Cc)
    areaBB=(PC[:,0].max()-PC[:,0].min())*(PC[:,1].max()-PC[:,1].min())
    Meta.append((c,(areaR/areaBB)))
  Meta.sort(key=lambda x: x[1], reverse=True)
  return Meta[0][0]

def cont2Img(img,Cont):
  stc=np.full(img.shape, False)
  stc[Cont.astype('int')[:,0],Cont.astype('int')[:,1]]=True
  return stc

def applyMask(img,binary):
  con3=np.zeros(img.shape)
  con3[:,:,0]=binary
  con3[:,:,1]=binary
  con3[:,:,2]=binary
  ROI1=np.int_(img)*con3
  return np.uint8(ROI1)

def stickerDetection(img,th=0.8):
  val_img=prePro(img)
  binary_img = get_binary(val_img,th)
  contours=getBiggestCont(binary_img,n=10)
  contours=closedCon(contours,binary_img)
  cont=contenedContours(contours)
  conte=nonContened(contours,cont)
  contours=containerContours(conte,cont)
  pca = PCA(n_components=2)
  selected_forms=similarForms(contours,pca)
  stc=cont2Img(binary_img,selected_forms)
  con=convex_hull_image(stc)
  return applyMask(img,con)

def stickerFilter(img,th=0.8):
  val_img=prePro(img)
  binary_img = get_binary(val_img,th)
  contours=getBiggestCont(binary_img,n=10)
  contours=closedCon(contours,binary_img)
  cont=contenedContours(contours)
  conte=nonContened(contours,cont)
  contours=containerContours(conte,cont)
  pca = PCA(n_components=2)
  selected_forms=similarForms(contours,pca)
  stc=cont2Img(binary_img,selected_forms)
  con=convex_hull_image(stc)
  return applyMask(img,np.invert(con))

def stickerCut(img,th=0.8):
  val_img=prePro(img)
  binary_img = get_binary(val_img,th)
  contours=getBiggestCont(binary_img,n=10)
  contours=closedCon(contours,binary_img)
  cont=contenedContours(contours)
  conte=nonContened(contours,cont)
  contours=containerContours(conte,cont)
  pca = PCA(n_components=2)
  selected_forms=similarForms(contours,pca)
  stc=cont2Img(binary_img,selected_forms)
  con=convex_hull_image(stc)
  pth=300
  Recorte_aumento=500
  TIndex=np.where(con==True)
  Xmax=np.max(TIndex[0])+Recorte_aumento
  Xmin=np.min(TIndex[0])-Recorte_aumento
  Ymax=np.max(TIndex[1])+Recorte_aumento
  Ymin=np.min(TIndex[1])-Recorte_aumento
  if Xmin<pth:
    Xmin=pth
  if Ymin<pth:
    Ymin=pth
  if Xmax>con.shape[0]-pth:
    Xmax=con.shape[0]-pth
  if Ymax>con.shape[1]-pth:
    Ymax=con.shape[1]-pth
  con[:,:]=False
  con[Xmin:Xmax,Ymin:Ymax]=True
  return applyMask(img,con)

def get_X_U(img,mask,n_segments=800):
    lum = color.rgb2gray(img)
    mask1=lum>0

    m_slic = slic(img, n_segments=n_segments,sigma=5,mask=mask)

    RID=set(m_slic.flatten())
    f=np.zeros((img.shape[0],img.shape[1],4))
    f[:,:,0:3]=img[:,:,0:3]
    f[:,:,3]=m_slic

    DIRID={i:{'U':np.zeros((3)),'X':np.zeros((3))} for i in RID}
    indx=np.where(f[:,:,3]==1)
    f[indx[0],indx[1],:]

    for i in RID:
        indx=np.where(f[:,:,3]==i)
        x=np.mean(f[indx[0],indx[1],:],axis=0)
        u=np.std(f[indx[0],indx[1],:],axis=0)
        DIRID[i]['X']=x
        DIRID[i]['U']=u
    return DIRID

def get_Statistical_Descriptors(img,n_segments=800):
    lum = color.rgb2gray(img)
    mask1=lum>0

    m_slic = slic(img, n_segments=n_segments,sigma=5,mask=mask1)

    RID=set(m_slic.flatten())
    f=np.zeros((img.shape[0],img.shape[1],4))
    f[:,:,0:3]=img[:,:,0:3]
    f[:,:,3]=m_slic

    DIRID={i:{'U':np.zeros((3)),'X':np.zeros((3)),
              'Per':np.zeros((3)),'Mo':np.zeros((3))} for i in RID}
    indx=np.where(f[:,:,3]==1)
    f[indx[0],indx[1],:]

    for i in RID:
        indx=np.where(f[:,:,3]==i)
        x=np.mean(f[indx[0],indx[1],:],axis=0)
        u=np.std(f[indx[0],indx[1],:],axis=0)
        perc=np.percentile(f[indx[0],indx[1],:],np.array([0,25,50,75,100]),axis=0)
        hist=np.histogram(f[indx[0],indx[1],:],bins=50,range=(0,255))
        Mo=hist[1][np.where(hist[0]==np.max(hist[0]))]
        DIRID[i]['X']=x
        DIRID[i]['U']=u
        DIRID[i]['Per']=perc
        DIRID[i]['Mo']=Mo
    return DIRID