close all;clear;clc;
myDir = uigetdir; %gets directory
myFiles = dir(fullfile(myDir,'*.jpg')); %gets all wav files in struct
values = readtable('db_sample_201901221525.csv');
sale = values{:, 2};
N_files = length(myFiles);
k = 764;
fid = fopen('validcrop.txt', 'a+');
while k <= length(myFiles)
    baseFileName = myFiles(k).name;
    fullFileName = fullfile(myDir, baseFileName);
    fprintf(1, 'Now reading %s\n', fullFileName);
    disp(k);
    a = imread(fullFileName);
    [eye, rect2] = imcrop(a);
    [nH, nW, ~] = size(eye);
    e_ntsc = rgb2ntsc(eye);
    e_ycbcr = rgb2ycbcr(eye);
    Cr = im2double(histeq(e_ycbcr(:,:,3)));
    T = histeq(e_ntsc(:,:,2));
    SC = histeq(e_ntsc(:,:,3));
    pruebex = Cr.*T.*SC;
    mp = mean2(pruebex);
    sp = std2(pruebex);
    mmask =pruebex>mp+sp;
    se = strel('disk',100);
    mask = imdilate(mmask, se);
    mmask = imfill(mmask,4, 'holes');
    bin = bwlabel(mmask);
    scx = round(nW/2);
    scy = round(2*nH/5);
    lv = 0;
    for i=1:nH
        if mmask(i, scx)==1 & i>scy
            lv = bin(i, scx);
            break
        end
    end
    rmmask = bin==lv;
    figure,
    imshow(bitand(im2uint8(rmmask), eye))
    opt = input('Save(s), Skip(k) or Retry(r)', 's');
    if strcmp(opt, 's')
        fprintf(fid, ' %.4f %s\n', rect2, baseFileName);
        fullFileName = fullfile('Sem_Auto', append('eye_', baseFileName));   
        imwrite(bitand(im2uint8(rmmask), eye), fullFileName);
    elseif strcmp(opt, 'k')
        
    elseif strcmp(opt, 'r')
        k =k-1;
    end
    close all;
    k = k+1;
end
fclose(fid);    
    
    
    
    
    
    