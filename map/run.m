% This is a demo code for our OCT denoising algorithm:
% Read Data
clear; close all

AllimgPath = '/Users/adam_lau/MIPAV/DeepLearning/denoise_traditional/map/';        % ??????????
imgDir  = dir([AllimgPath, '*.png']); % ????????jpg????????
time = [];
for i=1:length(imgDir)
    % choose which example sample to use
    imgPath = [AllimgPath, imgDir(i).name];
    % imgPath = '../Data/1_matching/0.png';
    % imgPath = '../Data/2_CycleGAN/0.png';
    savePath = ['/Users/adam_lau/MIPAV/DeepLearning/denoise_traditional/map/result/', imgDir(i).name];

    %%% the input image intensity should be within [0, 1]
    D_input = imread(imgPath);
    D_input = im2double(D_input(:,:,1));

    % create a colormap
    map = parula(1000); 
    map = map(1:999,:);
    map = [map;[1,1,1]];
    
    % Statistical parameter estimation
    alpha = estimatePar(D_input);

    % Run Proposed Method
    if ~exist('alpha','var')
        % default value
        alpha = 0.525;
    end
    % compute distribution coefficient
    c1 = (1-alpha^2/2)^(1/4);
    c2 = 1-(1-alpha^2/2)^(1/2);
    % parameter selection
    par.lambda = 0.4;
    par.gamma = 2;
    par.theta = 0.98;
    par.c1 = c1;
    par.c2 = c2;
    par.maxIter = 30;

    tic;
    [ U_ours_huberTV ] = ladexp_huberTV( D_input, par );
    toc;
    disp(['????????: ',num2str(toc)]);
    time(end+1)=toc;

    % Save images
    imwrite(U_ours_huberTV,savePath);
end
m1 = mean(time);
s1 = std(time);
