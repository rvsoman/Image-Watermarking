% clc; clear all; close all;
% host=imread('D:\MATLAB\bin\lena.jpg');
% [m n p]=size(host);
% subplot(1,3,1)
% imshow(host);
% title('Original Image');
% [host_LL,host_LH,host_HL,host_HH]=dwt2(host,'haar');
% water_mark=imread('fruits.jpg');
% water_mark=imresize(water_mark,[m n]);
% subplot(1,3,2)
% imshow(water_mark)
% title('Watermark');
% [water_mark_LL,water_mark_LH,water_mark_HL,water_mark_HH]=dwt2(water_mark,'haar');
% water_marked_LL = host_LL + (0.3*water_mark_LL);
% watermarked=idwt2(water_marked_LL,host_LH,host_HL,host_HH,'haar');
% subplot(1,3,3)
% imshow(uint8(watermarked));
% title('Watermarked image');
% imwrite(uint8(watermarked),'Watermarked.png');
% y='Watermarked succesfully';






% clc;clear all;close all;
% img  = imread('lena.jpg'); %Get the input image 
% img  = rgb2gray(img);      %Convert to grayscale image
% img  = double(img);
% c = 0.01; %Initialise the weight of Watermarking
% figure,imshow(uint8(img)),title('Original Image');
% [p q] = size(img);
% %Generate the key 
% n = awgn(img,4,3,'linear');
% N = imabsdiff(n,img);
% figure,imshow(double(N)),title('Key');
% [Lo_D,Hi_D,Lo_R,Hi_R] = wfilters('haar');%Obtain the fiters associated with haar
% [ca,ch,cv,cd] = dwt2(img,Lo_D,Hi_D);     %Compute 2D wavelet transform
% %Perform the watermarking
% y = [ca ch;cv cd];
% Y = y + c*abs(y).* N; 
% p=p/2;q=q/2;
% for i=1:p
%     for j=1:q
%         nca(i,j) = Y(i,j);
%         ncv(i,j) = Y(i+p,j);
%         nch(i,j) = Y(i,j+q);
%         ncd(i,j) =  Y (i+p,j+q);
%     end
% end
% %Display the Watermarked image
% wimg = idwt2(nca,nch,ncv,ncd,Lo_R,Hi_R);
% figure,imshow(uint8(wimg)),title('Watermarked Image');
% diff = imabsdiff(wimg,img);
% figure,imshow(double(diff));title('Differences');







clc; close all; clear all;
rgbimage=rgb2gray(imread('lena.jpg'));
[numrows numcols]=size(rgbimage);
[h_cA,h_cH,h_cV,h_cD]=dwt2(rgbimage,'haar');
% dec2d = [h_cA, h_cH; h_cV, h_cD];

%watermark image
rgbimage=rgb2gray(imread('forest.jfif'));
rgbimage = imresize(rgbimage,[numrows numcols]);
[w_cA,w_cH,w_cV,w_cD]=dwt2(rgbimage,'haar');

% dec2d = [w_cA, w_cH;  w_cV, w_cD ];
% watermarking
newhost_LL = (0.8*w_cA);
% newhost_LL = 0.9*w_LL;
% output
% rgb2=idwt2(newhost_LL,h_cH,h_cV,h_cD,'haar');
new_img=idwt2(h_cA,h_cH,h_cV,newhost_LL,'haar');
figure;imshow(uint8(new_img));title('Watermarked image');
% imwrite(uint8(rgb2),'Watermarked.jpg');






% % function [watermrkd_img,PSNR,IF,NCC,recmessage] = dwt(cover_object,message,k)
% h=msgbox('Processing');
% blocksize=8;
% %  message1 =message;
% % determine size of watermarked image
% Mc=size(cover_object,1);    %Height
% Nc=size(cover_object,2);    %Width
% Oc=size(cover_object,3);	%Width 
% max_message=Mc*Nc/(blocksize^2);
%  
% if (length(message) > max_message)
%     error('Message too large to fit in Cover Object')
% end
%  
% Mm=size(message,1);                         %Height
% Nm=size(message,2);                         %Width
% message=round(reshape(message,Mm*Nm,1)./256);
% message_vector=ones(1,max_message);
% % message_vector(1:length(message))=message;
%  message_vector=round(reshape(message,Mm*Nm,1)./256);
% % read in key for PN generator
% file_name='_key.bmp';
% key=double(imread(file_name))./256;
%  
% % reset MATLAB's PN generator to state "key"
% j = 1;
% for i =1:length(key)
% rand('state',key(i,j));
% end
%  
%  
% [cA1,cH1,cV1,cD1] = dwt2(cover_object,'haar');
%  % add pn sequences to H1 and V1 componants when message = 0 
% for (kk=1:length(message_vector))
%     pn_sequence_h=round(2*(rand(Mc/2,Nc/2,Oc)-0.5));
%     pn_sequence_v=round(2*(rand(Mc/2,Nc/2,Oc)-0.5));
%     
%     if (message(kk) == 0)
% %         cH1(:,:,1)=cH1(:,:,1)+k*pn_sequence_h;
% %         cV1(:,:,1)=cV1(:,:,1)+k*pn_sequence_v;
%         cH1=cH1+k*pn_sequence_h;
%         cV1=cV1+k*pn_sequence_v;
%     end
% end
% watermarked_image = idwt2(cA1,cH1,cV1,cD1,'haar',[Mc,Nc]); 
%  
% % convert back to uint8
% watermarked_image_uint8=uint8(watermarked_image);
%  
%  [message_vector,Mo,No] = dwtrecover(watermarked_image,k,message1);
%  recmessage=reshape(message_vector,Mo,No);
%  NCC=ncc(double(message1),recmessage);
% % calculate the PSNR
% I0     = double(cover_object);
% I1     = double(watermarked_image_uint8);
% Id     = (I0-I1);
% signal = sum(sum(I0.^2));
% noise  = sum(sum(Id.^2));
% MSE  = noise./numel(I0);
% peak = max(I0(:));
% PSNR = 10*log10(peak^2/MSE(:,:,1));
% % Image Fiedility
% IF = imfed(I0,Id);
% IF = mean(IF);
% %Normalized Cross Correlation
% watermrkd_img=watermarked_image_uint8;
% close(h) 
% % end
% 
