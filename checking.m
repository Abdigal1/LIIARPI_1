close all;clear;clc
myDir = uigetdir; %gets directory
myFiles = dir(fullfile(myDir,'*.jpg')); %gets all wav files in struct
for k = 1:length(myFiles)
  if k ==3
      break
  end
  baseFileName = myFiles(k).name;
  fullFileName = fullfile(myDir, baseFileName);
  a = imread(fullFileName);
  imshow(a);
  pause(1);
  close;
  fprintf(1, 'Now reading %s\n', fullFileName);
end