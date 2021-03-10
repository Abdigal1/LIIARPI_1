close all;clear;clc
myDir = uigetdir; %gets directory
myFiles = dir(fullfile(myDir,'*.jpg')); %gets all wav files in struct
values = readtable('db_sample_201901221525.csv');
sale = values{:, 2};
N_files = length(myFiles);
for k = 1:length(myFiles)
%   if k ==3
%       break
%   end
  %% Lee la imagen
  baseFileName = myFiles(k).name;
  fullFileName = fullfile(myDir, baseFileName);
  a = imread(fullFileName);
  %% 
  fprintf(1, 'Now reading %s %d/%d\n', fullFileName, k, N_files);
  %% Compara string
  idx = find(strcmp(sale, baseFileName(4:end)));
  ane_level = values{idx, 4};
  %% Start process
    b = rgb2hsv(a);
    [W, H, ~] = size(a);
    %Sin flash
    flag = true;
    mask = 144<=a(:,:,1) & 177>= a(:,:,1) & 149<=a(:,:,2)...
        &191>=a(:,:,2)& 152<=a(:,:,3) & 183>=a(:,:,3);
    mask = imfill(mask,4, 'holes');
    if sum(sum(mask))>0.15*W*H
        flag = false;
        %disp('Caso 2')%aumenta un poco 166 206
        mask = 145<=a(:,:,1) & 167>= a(:,:,1) & 182<=a(:,:,2)...
            &222>=a(:,:,2)& 198<=a(:,:,3) & 235>=a(:,:,3);
        mask = imfill(mask,4, 'holes');
    end
    if flag
        mask = mask & 0.5<=b(:,:,1) & 0.55>=b(:,:,1);
    end
    se = strel('line',25,25);
    mask = imdilate(mask, se);
    mask = imfill(mask,4, 'holes');
    cc = bwconncomp(mask,4);
    P=cc.NumObjects;
    defectdata = regionprops(cc, 'all');
    [~,idx]=sort([defectdata.Area]);
    defectdata=defectdata(idx);
    if isempty(defectdata)
        continue
    end
    firstb = defectdata(end).BoundingBox;
    scx = round(defectdata(end).Centroid(1));
    scy = round(defectdata(end).Centroid(2));
    x = firstb(1);
    y = firstb(2);
    ww = firstb(3);
    hh = firstb(4);
    N = 2.2;
    nx = max(0,floor(x-(ww/2)*(N/2)));
    ny = max(0,floor(y-(hh/2)*(N/2)));
    rb = [nx ny min(H-nx, N*ww) min(W-ny, 1.1*N*hh)];
    scx = scx-nx;
    scy = scy-ny;
    eye = imcrop(a, rb);
    [nH, nW, ~] = size(eye);
    if (nH*nW/(H*W))>=0.02
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
        lv = 0;
        for i=1:nH
            if mmask(i, scx)==1 & i>scy
                lv = bin(i, scx);
                break
            end
        end
        rmmask = bin==lv;
        if sum(sum(rmmask))>= 15000
  %% End process
          if ane_level >5 && ane_level <20
              %% Guarda en la nueva carpeta
              fullFileName = fullfile('Processed', append('eye_', baseFileName(4:end)));   
              imwrite(bitand(im2uint8(rmmask), eye), fullFileName);
          end
        end
    end
end
%sumen 15000
%relacion are 0.02


