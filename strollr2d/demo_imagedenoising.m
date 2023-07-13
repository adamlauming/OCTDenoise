clear;

AllimgPath = '/Users/adam_lau/MIPAV/DeepLearning/denoise_traditional/strollr2d/';    
imgDir  = dir([AllimgPath, '*.png']); 
time = [];
for i=1:length(imgDir)
    % choose which example sample to use
    imgPath = [AllimgPath, imgDir(i).name];
    D_input = imread(imgPath);
    noisy = im2double(D_input(:,:,1));
    oracle = noisy;
    savePath = ['/Users/adam_lau/MIPAV/DeepLearning/denoise_traditional/strollr2d/result/', imgDir(i).name];

    %%%%%%%%%%%%%%%% strollr2D image denoising demo %%%%%%%%%%%%%%%%%%%%%
    data.noisy      =   noisy;
    data.oracle     =   oracle;
    param.sig       =   20;                % sig = 20
    param           =   getParam_icassp2017(param);
    tic;
    [Xr, psnrXr]= strollr2d_imagedenoising(data, param);
    toc;
    disp(['res: ',num2str(toc)]);
    time(end+1)=toc;
    imwrite(Xr,savePath);
    fprintf( 'STROLLR-2D denoising completes! \n Denoised PSNR = %2.2f \n', psnrXr);    
end
m1 = mean(time);
s1 = std(time);