close all;clear;
a = imread('c1anemia-152.jpg');%Falta 101 subir 664
%Caso extremo 676
b = rgb2hsv(a);
[W, H, ~] = size(a);
%Sin flash
flag = true;
mask = 144<=a(:,:,1) & 177>= a(:,:,1) & 149<=a(:,:,2)...
    &191>=a(:,:,2)& 152<=a(:,:,3) & 183>=a(:,:,3);
mask = imfill(mask,4, 'holes');
if sum(sum(mask))>0.15*W*H
    flag = false;
    disp('Caso 2')%aumenta un poco 166 206
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
%Con flash

figure, imshow(mask);
figure, imshow([bitand(im2uint8(mask), a), a]);


cc = bwconncomp(mask,4);
P=cc.NumObjects;
defectdata = regionprops(cc, 'all');
[~,idx]=sort([defectdata.Area]);
defectdata=defectdata(idx);
firstb = defectdata(end).BoundingBox;
x = firstb(1);
y = firstb(2);
ww = firstb(3);
hh = firstb(4);
nx = max(0,floor(x-ww/2));
ny = max(0,floor(y-hh/2));
rb = [nx ny min(H-nx, 2*ww) min(W-ny, 2*hh)];
figure, imshow(imcrop(a, rb))