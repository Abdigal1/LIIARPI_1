close all;clear;
a = imread('c1anemia-238.jpg');

%Sin flash
%mask = 144<=a(:,:,1) & 177>= a(:,:,1) & 149<=a(:,:,2) &191>=a(:,:,2)& 152<=a(:,:,3) & 183>=a(:,:,3);
%Con flash
mask = 145<=a(:,:,1) & 167>= a(:,:,1) & 182<=a(:,:,2) &222>=a(:,:,2)& 198<=a(:,:,3) & 235>=a(:,:,3);
mask = imfill(mask,4, 'holes');
figure, imshow(mask);
figure, imshow([bitand(im2uint8(mask), a), a]);