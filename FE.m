filename = 't2.jpg';

Image_orig=imread(filename);

try
    Image_orig=rgb2gray(Image_orig);
catch
end

subplot(1,3,1)
imshow(Image_orig)
title('(a) Original Image')


Image_orig=double(Image_orig);

Image_orig_theshold = Image_orig;

gt = graythresh(Image_orig_theshold);
edgeThreshold = gt;

handles.LPF = round(gt/2, 2); 
handles.Phase_strength= 0.4;
handles.Warp_strength = 2; 
handles.Thresh_min=-1;      
handles.Thresh_max=0.0019;  
Morph_flag = 1 ; 
[Edge, PST_Kernel]= PST(Image_orig,handles,Morph_flag);

if Morph_flag ==0
    subplot(1,3,2)
    imshow(Edge/max(max(Edge))*3)
    title('(b) Detected features using MPST')
    file_and_extension=strsplit(filename,'.');
    output_path=char(strcat('../Test_Images/',file_and_extension(1),'_edge.tif'));
    imwrite(Edge/max(max(Edge))*3,output_path);
else
    subplot(1,3,2)
    imshow(Edge)
    title('(b) Detected features using MPST')
    file_and_extension=strsplit(filename,'.');
    output_path=char(strcat('../Test_Images/',file_and_extension(1),'_edge.jpg'));
    imwrite(Edge,output_path);
    overlay = double(imoverlay(Image_orig, Edge/1000000, [1 0 0]));
    subplot(1,3,3)
    imshow(overlay/max(max(max(overlay))));
    title({'(c) MPST features overlaid', 'with Original Image'});
    output_path=char(strcat('../Test_Images/',file_and_extension(1),'_overlay.jpg'));
    imwrite(overlay/max(max(max(overlay))),output_path);
end

figure
[D_PST_Kernel_x, D_PST_Kernel_y]=gradient(PST_Kernel);
mesh(sqrt(D_PST_Kernel_x.^2+D_PST_Kernel_y.^2))

title_D = sprintf('(d) MPST Kernel Phase Gradient Profiles with W = %.0f and S = %.2f', handles.Warp_strength, handles.Phase_strength);
title(title_D)
image_PST = imread('t2_overlay.tif');
seg_img = Image_orig;
[~, ~, ind] = size(seg_img);
if (ind == 3)
    seg_img = rgb2gray(seg_img);
end
x = double(seg_img);
signal1 = seg_img(:,:);

[cA1,cH1,cV1,cD1] = dwt2(signal1,'db4');
[cA2,cH2,cV2,cD2] = dwt2(cA1,'db4');
[cA3,cH3,cV3,cD3] = dwt2(cA2,'db4');
[cA4,cH4,cV4,cD4] = dwt2(cA3,'db4');
DWT_feat = [cA4,cH4,cV4,cD4];
[img_rows, img_cols, img_ColorChannels] = size(Image_orig);
fprintf('\n The size of imput image is %d X %d with %d Color Channels\n', img_rows, img_cols, img_ColorChannels);
[img_rows_dwt, img_cols_dwt, img_ColorChannels_dwt] = size(DWT_feat);
fprintf('\n The size of FE image is %d X %d with %d Color Channels\n', img_rows_dwt, img_cols_dwt, img_ColorChannels_dwt);
G = DWT_feat;
g = graycomatrix(G);
stats = graycoprops(g,'Contrast Correlation Energy Homogeneity');
Contrast = stats.Contrast;
Correlation = stats.Correlation;
Energy = stats.Energy;
Homogeneity = stats.Homogeneity;
Mean = mean2(G);
Standard_Deviation = std2(G);
Entropy = entropy(G);
RMS = mean2(rms(G));
Skewness = skewness(img);
Variance = mean2(var(double(G)));

a = sum(double(G(:)));
Smoothness = 1-(1/(1+a));
Kurtosis = kurtosis(double(G(:)));
Skewness = skewness(double(G(:)));
m = size(G,1);
n = size(G,2);
in_diff = 0;
for i = 1:m
    for j = 1:n
        temp = G(i,j)./(1+(i-j).^2);
        in_diff = in_diff+temp;
    end
end
IDM = double(in_diff);
    
feat = [Contrast,Correlation,Energy,Homogeneity, Mean, Standard_Deviation, Entropy, RMS, Variance, Smoothness, Kurtosis, Skewness, IDM];


fprintf('\n\n1. The value of Contrast is %0.5f', Contrast);
fprintf('\n2. The value of Correlation is %0.5f', Correlation);
fprintf('\n3. The value of Energy is %0.5f', Energy);
fprintf('\n4. The value of Homogeneity is %0.5f', Homogeneity);
fprintf('\n5. The value of Mean is %0.5f', Mean);
fprintf('\n6. The value of Standard_Deviation is %0.5f', Standard_Deviation);
fprintf('\n7. The value of Entropy is %0.5f', Entropy);
fprintf('\n8. The value of RMS is %0.5f', RMS);
fprintf('\n9. The value of Variance is %0.5f', Variance);
fprintf('\n10. The value of Smoothness is %0.5f', Smoothness);
fprintf('\n11. The value of Kurtosis is %0.5f', Kurtosis);
fprintf('\n12. The value of Skewness is %0.5f', Skewness);
fprintf('\n13. The value of IDM is %0.5f', IDM);
fprintf('\n');
fprintf('=========================================');
load Trainset.mat
xdata = meas;
group = label;
SVMModel = fitcsvm(xdata,group,'Standardize',true,'BoxConstraint', 5, 'KernelFunction', 'polynomial');
classes = predict(SVMModel,xdata);
cp = classperf(classes,group);
fprintf('\n\nCorrect Rate value is : %3.3f \n', cp.CorrectRate * 100);

load Trainset.mat
data = meas;
groups = ismember(label,'BENIGN   ');
groups = ismember(label,'MALIGNANT');
[train,test] = crossvalind('HoldOut',groups);
cp = classperf(groups);
svmStruct = fitcsvm(data(train,:),groups(train),'Standardize',true,'BoxConstraint', 50,'KernelFunction','linear');
classes = predict(svmStruct,data(test,:));
classperf(cp,classes,test);
Accuracy_Classification = cp.CorrectRate.*100;
fprintf('\n\nAccuracy of Linear kernel is: %3.3f',Accuracy_Classification);
svmStruct_RBF = fitcsvm(data(train,:),groups(train),'Standardize',true,'BoxConstraint', 50,'KernelFunction','rbf');
classes2 = predict(svmStruct_RBF,data(test,:));
classperf(cp,classes2,test);
Accuracy_Classification_RBF = cp.CorrectRate.*100;
fprintf('\n\nAccuracy of RBF kernel is: %3.3f%',Accuracy_Classification_RBF);
svmStruct_Poly = fitcsvm(data(train,:),groups(train),'Standardize',true,'BoxConstraint', 50,'KernelFunction','polynomial');
classes3 = predict(svmStruct_Poly,data(test,:));
classperf(cp,classes3,test);
Accuracy_Classification_Poly = cp.CorrectRate.*100;
fprintf('\n\nAccuracy of Polynomial kernel is: %3.3f %',Accuracy_Classification_Poly);
fprintf('\n\n');
