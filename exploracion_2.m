close all;clear;
%% Detecci�n del ojo(eye)
a = imread('c1anemia-676.jpg');%Falta 101 166 572
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
figure, imshow(eye);


%% Detecci�n de la conjuntiva

% subplot(211), imshow(eye);
% title('Original');
% subplot(212), imshow(rgb2gray(eye));
% title('Grises');
% figure,
% subplot(221), imshow(eye);
% title('Original');
e_hsv = rgb2hsv(eye);
% subplot(222), imshow(e_hsv(:,:,1));
% title('H');
% subplot(223), imshow(e_hsv(:,:,2));
% title('S');
% subplot(224), imshow(e_hsv(:,:,3));
% title('V');
% 
% figure,
% subplot(221), imshow(eye);
% title('Original');
e_lab = rgb2lab(eye);
% subplot(222), imshow(e_lab(:,:,1));
% title('L');
% subplot(223), imshow(e_lab(:,:,2));
% title('A');
% subplot(224), imshow(e_lab(:,:,3));
% title('B');
% 
% figure,
% subplot(221), imshow(eye);
% title('Original');
e_ntsc = rgb2ntsc(eye);
% subplot(222), imshow(e_ntsc(:,:,1));
% title('N');
% subplot(223), imshow(e_ntsc(:,:,2));
% title('T');
% subplot(224), imshow(e_ntsc(:,:,3));
% title('SC');
% 
% figure,
% subplot(221), imshow(eye);
% title('Original');
e_xyz = rgb2xyz(eye);
% subplot(222), imshow(e_xyz(:,:,1));
% title('X');
% subplot(223), imshow(e_xyz(:,:,2));
% title('Y');
% subplot(224), imshow(e_xyz(:,:,3));
% title('Z');
% 
% figure,
% subplot(221), imshow(eye);
% title('Original');
e_ycbcr = rgb2ycbcr(eye);
% subplot(222), imshow(e_ycbcr(:,:,1));
% title('Y');
% subplot(223), imshow(e_ycbcr(:,:,2));
% title('Cb');
% subplot(224), imshow(e_ycbcr(:,:,3));
% title('Cr');
%% Mas pruebas
Cr = im2double(histeq(e_ycbcr(:,:,3)));
T = histeq(e_ntsc(:,:,2));
SC = histeq(e_ntsc(:,:,3));
% Cr(talvez) T(talvez) SC(talvez) S(tela)
% figure,
pruebex = Cr.*T.*SC;
% subplot(221), imshow(pruebex);
% title('Cr.*T.*SC');
% subplot(222), imshow(Cr.*T);
% title('Cr.*T');
% subplot(223), imshow(Cr.*SC);
% title('Cr.*SC');
% subplot(224), imshow(T.*SC);
% title('T.*SC');
% close all;
%% Aea

mp = mean2(pruebex);
sp = std2(pruebex);
mmask =pruebex>mp+sp;
se = strel('disk',100);
mask = imdilate(mmask, se);
mmask = imfill(mmask,4, 'holes');
debug = bitand(im2uint8(mmask), eye);
debug = insertShape(debug,'circle',[scx scy 35],'LineWidth',5);
figure, imshow(debug)

%% Final mascara
bin = bwlabel(mmask);
[nH, nW, ~] = size(eye);
lv = 0;
for i=1:nH
    if mmask(i, scx)==1 & i>scy
        lv = bin(i, scx);
        disp(scx);
        disp(i);
        break
    end
end
rmmask = bin==lv;
figure, imshow(rmmask)




