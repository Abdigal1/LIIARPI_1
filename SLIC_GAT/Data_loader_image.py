from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

def read_csv(path,n):
      landmarks_frame = pd.read_csv(path)
      img_name = landmarks_frame.iloc[n, 0]
      landmarks = landmarks_frame.iloc[n, 1:]
      landmarks = np.asarray(landmarks)
      landmarks = landmarks.astype('float').reshape(-1, 2)
      return landmarks


def show_landmarks(image, landmarks):
      plt.imshow(image)
      plt.scatter(landmarks[:, 0], landmarks[:, 1], s=10, marker='.', c='r')
      plt.pause(0.001)  # pause a bit so that plots are updated

#plt.figure()
#show_landmarks(io.imread(os.path.join('faces/', img_name)),
#               landmarks)
#plt.show()

class Dataset(torch.utils.data.Dataset):
      def __init__(self, csv_file, root_dir,transform=None):
            'Initialization'
            self.root_dir = root_dir

            raw_data=np.array(os.listdir(root_dir))
            y=np.vectorize(pyfunc=lambda stg:stg.split(".")[0].split("_")[-1])(raw_data).reshape(-1,1)
            self.landmarks_frame=np.hstack((raw_data.reshape(-1,1),y))
            self.transform = transform

      def __len__(self):
            'Denotes the total number of samples'
            return len(self.landmarks_frame)

      def __getitem__(self, idx):
            'Generates one sample of data'
            # Select sample
            if torch.is_tensor(idx):
                  idx=idx.tolist()

            img_name = os.path.join(self.root_dir,self.landmarks_frame[idx, 0])
            image = io.imread(img_name)
            landmarks = self.landmarks_frame[idx, 1:]
            landmarks = np.array([landmarks])

            landmarks = landmarks.astype('float')
            sample = {'image': image, 'landmarks': landmarks}
            if self.transform:
                  sample=self.transform(sample)

            return sample

class Dataset_direct(torch.utils.data.Dataset):
      def __init__(self,root_dir,transform=None):
            'Initialization'
            self.landmarks_frame = np.array(os.listdir(root_dir))
            self.transform = transform

      def __len__(self):
            'Denotes the total number of samples'
            return len(self.landmarks_frame)

      def __getitem__(self, idx):
            'Generates one sample of data'
            # Select sample
            if torch.is_tensor(idx):
                  idx=idx.tolist()

            img_name = os.path.join(self.root_dir,self.landmarks_frame.iloc[idx, 0])
            image = io.imread(img_name)
            landmarks = self.landmarks_frame.iloc[idx, 1:]
            landmarks = np.array([landmarks])

            landmarks = landmarks.astype('float').reshape(-1, 2)
            sample = {'image': image, 'landmarks': landmarks}
            if self.transform:
                  sample=self.transform(sample)

            return sample


class Rescale(object):
      def __init__(self, output_size):
            assert isinstance(output_size, (int, tuple))
            self.output_size = output_size

      def __call__(self, sample):
            image, landmarks = sample['image'], sample['landmarks']

            h, w = image.shape[:2]
            if isinstance(self.output_size, int):
                  if h > w:
                        new_h, new_w = self.output_size * h / w, self.output_size
                  else:
                        new_h, new_w = self.output_size, self.output_size * w / h
            else:
                  new_h, new_w = self.output_size

            new_h, new_w = int(new_h), int(new_w)

            img = transform.resize(image, (new_h, new_w))

        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively
            landmarks = landmarks * [new_w / w, new_h / h]

            return {'image': img, 'landmarks': landmarks}

class RandomCrop(object):
      def __init__(self, output_size):
            assert isinstance(output_size, (int, tuple))
            if isinstance(output_size, int):
                  self.output_size = (output_size, output_size)
            else:
                  assert len(output_size) == 2
                  self.output_size = output_size

      def __call__(self, sample):
            image, landmarks = sample['image'], sample['landmarks']

            h, w = image.shape[:2]
            new_h, new_w = self.output_size

            top = np.random.randint(0, h - new_h)
            left = np.random.randint(0, w - new_w)

            image = image[top: top + new_h,
                      left: left + new_w]

            landmarks = landmarks - [left, top]

            return {'image': image, 'landmarks': landmarks}

class ToTensor(object):
      def __call__(self, sample):
            image, landmarks = sample['image'], sample['landmarks']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
            image = image.transpose((2, 0, 1))
            return {'image': torch.from_numpy(image),
                'landmarks': torch.from_numpy(landmarks)}

#class applySLIC(object):

def show_landmarks_batch(sample_batched):

      images_batch, landmarks_batch = sample_batched['image'], sample_batched['landmarks']
      batch_size = len(images_batch)
      im_size = images_batch.size(2)      
      grid = utils.make_grid(images_batch)
      plt.imshow(grid.numpy().transpose((1, 2, 0)))

      for i in range(batch_size):
            plt.scatter(landmarks_batch[i, :, 0].numpy() + i * im_size,
                  landmarks_batch[i, :, 1].numpy(),
                  s=10, marker='.', c='r')

      plt.title('Batch from dataloader')