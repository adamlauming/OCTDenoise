clear all;
close all;
clc;

%% calculate the CNR and SNR
ave_img = imread('image_strollr_3.png');
img1 = imread('image_base3.png');
img2 = imread('image_strollr_3.png');
%     ave_img = rgb2gray(ave_img);
bscan = 1; 
imagesc(ave_img);colormap(gray)
% draw the bg_region and roi_regions 
hold on
x0=350;y0=500;width0=200;height0=50;
rectangle('Position',[x0,y0,width0,height0],'LineWidth',1,'EdgeColor','b');
regionbg = ave_img(500:550,350:550);
ROI_num = 5;
x1=65;y1=620;width1=20;height1=20;
rectangle('Position',[x1,y1,width1,height1],'LineWidth',1,'EdgeColor','b');
region1 = ave_img(621:640,66:85);
x2=250;y2=710;width2=20;height2=20;
rectangle('Position',[x2,y2,width2,height2],'LineWidth',1,'EdgeColor','b');
region2 = ave_img(711:730,251:270);
x3=450;y3=790;width3=20;height3=20;
rectangle('Position',[x3,y3,width3,height3],'LineWidth',1,'EdgeColor','b');
region3 = ave_img(791:810,451:470);
x4=520;y4=705;width4=20;height4=20;
rectangle('Position',[x4,y4,width4,height4],'LineWidth',1,'EdgeColor','b');
region4 = ave_img(706:725,521:540);
x5=870;y5=550;width5=20;height5=20;
rectangle('Position',[x5,y5,width5,height5],'LineWidth',1,'EdgeColor','b');
region5 = ave_img(551:570,871:890);

% x0=400;y0=500;width0=200;height0=50;
% rectangle('Position',[x0,y0,width0,height0],'LineWidth',2,'EdgeColor','r');
% regionbg = ave_img(500:550,400:600);
% ROI_num = 5;
% x1=20;y1=600;width1=20;height1=20;
% rectangle('Position',[x1,y1,width1,height1],'LineWidth',2,'EdgeColor','r');
% region1 = ave_img(601:620,21:40);
% x2=280;y2=620;width2=20;height2=20;
% rectangle('Position',[x2,y2,width2,height2],'LineWidth',2,'EdgeColor','r');
% region2 = ave_img(621:640,281:300);
% x3=680;y3=600;width3=20;height3=20;
% rectangle('Position',[x3,y3,width3,height3],'LineWidth',2,'EdgeColor','r');
% region3 = ave_img(601:620,681:700);
% x4=520;y4=705;width4=20;height4=20;
% rectangle('Position',[x4,y4,width4,height4],'LineWidth',2,'EdgeColor','r');
% region4 = ave_img(706:725,521:540);
% x5=780;y5=600;width5=20;height5=20;
% rectangle('Position',[x5,y5,width5,height5],'LineWidth',2,'EdgeColor','r');
% region5 = ave_img(601:620,781:800);

% x0=430;y0=470;width0=200;height0=50;
% rectangle('Position',[x0,y0,width0,height0],'LineWidth',1,'EdgeColor','b');
% regionbg = ave_img(470:520,430:630);
% ROI_num = 5;
% x1=300;y1=420;width1=20;height1=20;
% rectangle('Position',[x1,y1,width1,height1],'LineWidth',1,'EdgeColor','b');
% region1 = ave_img(421:440,301:320);
% x2=420;y2=560;width2=20;height2=20;
% rectangle('Position',[x2,y2,width2,height2],'LineWidth',1,'EdgeColor','b');
% region2 = ave_img(561:580,421:440);
% x3=520;y3=680;width3=20;height3=20;
% rectangle('Position',[x3,y3,width3,height3],'LineWidth',1,'EdgeColor','b');
% region3 = ave_img(681:700,521:540);
% x4=600;y4=580;width4=20;height4=20;
% rectangle('Position',[x4,y4,width4,height4],'LineWidth',1,'EdgeColor','b');
% region4 = ave_img(581:600,601:620);
% x5=670;y5=505;width5=20;height5=20;
% rectangle('Position',[x5,y5,width5,height5],'LineWidth',1,'EdgeColor','b');
% region5 = ave_img(671:690,506:525);

% calculate the SNR
ave_bg = mean(regionbg);
ave_bg = mean(ave_bg); % calculate the mean
std_bg = std2(regionbg); % calculate the standard deviation
SNR(bscan) = ave_bg/std_bg; % SNR = I/sigema;
SNR_normal(bscan) = SNR(bscan); % normalization
fprintf('the SNR of the %d image(s) is = %f \n',bscan,SNR_normal(bscan))

% calculate the CNR
ave_roi_1 = mean(mean(region1));
std_roi_1 = std2(region1);
CNRpara_1 = (ave_roi_1 - ave_bg)/sqrt(std_roi_1.^2 + std_bg.^2);
ave_roi_2 = mean(mean(region2));
std_roi_2 = std2(region2);
CNRpara_2 = (ave_roi_2 - ave_bg)/sqrt(std_roi_2.^2 + std_bg.^2);
ave_roi_3 = mean(mean(region3));
std_roi_3 = std2(region3);
CNRpara_3 = (ave_roi_3 - ave_bg)/sqrt(std_roi_3.^2 + std_bg.^2);
ave_roi_4 = mean(mean(region4));
std_roi_4 = std2(region4);
CNRpara_4 = (ave_roi_4 - ave_bg)/sqrt(std_roi_4.^2 + std_bg.^2);
ave_roi_5 = mean(mean(region5));
std_roi_5 = std2(region5);
CNRpara_5 = (ave_roi_5 - ave_bg)/sqrt(std_roi_5.^2 + std_bg.^2);
CNR(bscan) = (CNRpara_1 + CNRpara_2 + CNRpara_3 + CNRpara_4 + CNRpara_5)/5;
fprintf('the CNR of the %d image(s) is = %f \n',bscan,CNR(bscan))

mse = metrix_mse(img1, img2)
psnr = metrix_psnr(img1, img2)
[mssim, ssim_map] = ssim(img1, img2);
mssim
ENL = (ave_roi_1^2/std_roi_1^2 + ave_roi_2^2/std_roi_2^2 + ave_roi_3^2/std_roi_3^2 + ave_roi_4^2/std_roi_4^2 + ave_roi_5^2/std_roi_5^2) /5