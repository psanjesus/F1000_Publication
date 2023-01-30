function ImgEnhancement

image = imread('w1.jpg');

[~, ~, ind] = size(image);
if (ind == 3)
    image = rgb2gray(image);
end

subplot(2,7,1); imshow(image); title('(a) Input Image');

subplot(2,7,8); imhist(image); title({'(h) Histogram:','Input Image'});

LocalHistogramEqualization(image);

GlobalHistogramEqualization(image);
 
subplot(2,7,4);
AHE_image = adapthisteq(image);
imshow(AHE_image);title('(d) AHE');

subplot(2,7,11);
imhist(AHE_image); title('(k) Histogram: AHE');

fprintf('\n');
fprintf('The following are the w.r.t to A H E: \n');
PSNR_AHE_image = psnr(image, AHE_image);
fprintf('The Peak-SNR value of NMHE is %0.4f\n', PSNR_AHE_image);

[~, MSE_AHE_image] = measerr(image, AHE_image);
fprintf('The MSE value of NMHE is: %.5g\n', MSE_AHE_image/100);

[ssimval_AHE_image, ~] = ssim(image, AHE_image);
fprintf('The SSIM value of NMHE is %0.4f\n', ssimval_AHE_image);

subplot(2,7,5);
CLAHE_image = adapthisteq(image,'clipLimit',0.02,'Distribution','rayleigh');
imshow(CLAHE_image);title('(e) CLAHE');

subplot(2,7,12);
imhist(CLAHE_image); title('(l) Histogram: CLAHE');

fprintf('\n');
fprintf('The following are the w.r.t to C L A H E: \n');
PSNR_CLAHE_image = psnr(image, CLAHE_image);
fprintf('The Peak-SNR value of NMHE is %0.4f\n', PSNR_CLAHE_image);

[~, MSE_CLAHE_image] = measerr(image, CLAHE_image);
fprintf('The MSE value of NMHE is: %.5g\n', MSE_CLAHE_image/100);

[ssimval_CLAHE_image, ~] = ssim(image, CLAHE_image);
fprintf('The SSIM value of NMHE is %0.4f\n', ssimval_CLAHE_image);

nmHE = NMHE_5(image);
subplot(2,7,6); imshow(nmHE); title('(f) NMHE');

nmHE_HE = histeq(nmHE);
subplot(2,7,13); imhist(nmHE_HE);title('(m) Histogram: NMHE');

fprintf('\n');
fprintf('The following are the w.r.t to N M H E: \n');
PSNR_nmHE = psnr(image, nmHE);
fprintf('The Peak-SNR value of NMHE is %0.4f\n', PSNR_nmHE);

[~, MSE_nmHE] = measerr(image, nmHE);
fprintf('The MSE value of NMHE is: %.5g\n', MSE_nmHE/100);

[ssimval, ~] = ssim(image, nmHE);
fprintf('The SSIM value of NMHE is %0.4f\n', ssimval);

DynamicHistogramEqualization(image);
