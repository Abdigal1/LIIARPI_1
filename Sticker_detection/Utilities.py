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

  