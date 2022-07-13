% Hashna Rasheed COURSE WORK
%Close all open figures
close all
%Clear the workspace
clear
%Clear the command window
clc
%% 
imds=imageDatastore('natural_images',...
    'IncludeSubfolders',true,...
    'LabelSource','foldernames');
%% 
table=countEachLabel(imds)
%% 
minSetCount = min(table{:,2});

%% 
imds = splitEachLabel(imds, minSetCount, 'randomize');
countEachLabel(imds)

%% 
[imdsTrain,imdsValidation] = splitEachLabel(imds,0.7 ,'randomize');
%% 
numDisplay=30;
numTrain=numel(imdsTrain.Files);
% the randperm function returns a randomized permutation of integers
idx=randperm(numTrain,numDisplay);
% loop over the index
figure
for i=1:numDisplay
   Input=readimage(imdsTrain,idx(i));
   class=cellstr(imdsTrain.Labels(idx(i)));
   subplot(6,6,i)
   imshow(Input);title(class{1})
end
%% 
augmenter = imageDataAugmenter('RandXReflection', true);
augimdsTrain = augmentedImageDatastore([200 200], imdsTrain,'ColorPreprocessing', 'gray2rgb', 'DataAugmentation', augmenter);
augimdsValidation = augmentedImageDatastore([200 200], imdsValidation,'ColorPreprocessing', 'gray2rgb', 'DataAugmentation', augmenter);
%% 
layers = [imageInputLayer([200 200 3])
convolution2dLayer(3,16,'Padding','same')
batchNormalizationLayer
reluLayer
maxPooling2dLayer(2,'Stride',2)
convolution2dLayer(3,32,'Padding','same')
batchNormalizationLayer
reluLayer
maxPooling2dLayer(2,'Stride',2)
convolution2dLayer(3,64,'Padding','same')
batchNormalizationLayer
reluLayer
maxPooling2dLayer(2,'Stride',2)
convolution2dLayer(3,128,'Padding','same')
batchNormalizationLayer
reluLayer
maxPooling2dLayer(2,'Stride',2)
convolution2dLayer(3,256,'Padding','same')
batchNormalizationLayer
reluLayer
fullyConnectedLayer(8)
softmaxLayer
classificationLayer];
%% 
options = trainingOptions('sgdm', ...
 'InitialLearnRate',0.01, ...
 'MaxEpochs',10, ...
 'Shuffle','every-epoch', ...
 'ValidationData',augimdsValidation, ...
 'ValidationFrequency',30, ...
 'Verbose',false, ...
 'Plots','training-progress');
%% 
net = trainNetwork(augimdsTrain,layers,options);
%% 
YPred =classify(net,augimdsValidation);
Yvalidation=imdsValidation.Labels;
accuracy= sum(YPred ==Yvalidation)/numel(Yvalidation);
%% 
plotconfusion(Yvalidation,YPred )

