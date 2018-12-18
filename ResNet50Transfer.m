% Load ResNet50
  net = resnet50;  

%  net.Layers(1);
%  inputSize = net.Layers(1).InputSize;
% layers = net.Layers;
% for i = 1:10
%         if isprop(layers(i),'WeightLearnRateFactor')
%             layers(i).WeightLearnRateFactor = 0;
%         end
%         if isprop(layers(i),'WeightL2Factor')
%             layers(i).WeightL2Factor = 0;
%         end
%         if isprop(layers(i),'BiasLearnRateFactor')
%             layers(i).BiasLearnRateFactor = 0;
%         end
%         if isprop(layers(i),'BiasL2Factor')
%             layers(i).BiasL2Factor = 0;
%         end
%     end

% Change last layer from 1000 classes to 136
lgraph = layerGraph(net);
numClasses = 136;
lgraph = removeLayers(lgraph, {'fc1000','fc1000_softmax','ClassificationLayer_fc1000'}); 
newLayers = [ 
fullyConnectedLayer(numClasses,'Name','fc','WeightLearnRateFactor' ,10, 'BiasLearnRateFactor', 10 ) 
softmaxLayer('Name','softmax') 
classificationLayer('Name','classoutput')]; 
lgraph = addLayers(lgraph,newLayers); 
lgraph = connectLayers(lgraph,'avg_pool','fc');

%set up for training
allImages = imageDatastore('FinalImage2vgg' , 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
[imd1, imd2, imd3, imd4, imd5] = splitEachLabel(allImages,0.2,0.2,0.2,0.2,0.2,'randomize');
k = 5;
partStores{1} = imd1.Files ;
partStores{2} = imd2.Files ;
partStores{3} = imd3.Files ;
partStores{4} = imd4.Files ;
partStores{5} = imd5.Files ;
% partStores{6} = imd6.Files ;
% partStores{7} = imd7.Files ;
% partStores{8} = imd8.Files ;
% partStores{9} = imd9.Files ;
% partStores{10} = imd10.Files; 
% partStores{k} = [];
% s1 = shuffle(allImages)
% for i = 1:k
%    temp = partition(s1, k, i);
%    partStores{i} = temp.Files;
% end
idx = crossvalind('Kfold', k, k);


for i = 1:k
    i;
    test_idx = (idx == i);
    train_idx = ~test_idx;

    testImages = imageDatastore(partStores{test_idx}, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
    trainingImages = imageDatastore(cat(1, partStores{train_idx}), 'IncludeSubfolders', true, 'LabelSource', 'foldernames');

    opts = trainingOptions('sgdm','InitialLearnRate' , 0.001,'Plots','training-progress', 'Verbose',true,'VerboseFrequency',1, 'MaxEpochs', 10,'ValidationData',(testImages),'ValidationFrequency',32, 'MiniBatchSize',32, 'ExecutionEnvironment','gpu');
    myNet= trainNetwork(trainingImages,lgraph, opts);

    %Measure Network accuracy
    
    [predictedLabels,scores]  = classify(myNet, testImages, 'ExecutionEnvironment' ,'gpu');
    accuracy(i) = mean(predictedLabels == testImages.Labels);
    [confMat,order] = confusionmat(testImages.Labels, predictedLabels);
    confMatALL{i} = confMat;
    % confMat = confMat./sum(confMat,2);
    % mean(diag(confMat))
    disp(accuracy(i));
    
    %Calculate Recall
    for a =1:size(confMat,1)
        recall(a) = confMat(a,a)/sum(confMat(a,:));
    end
    recall(isnan(recall))=[0];
    Recall = sum(recall)/size(confMat,1);
    RecallALL(i) = Recall;
    disp(Recall);

    %Calculate Precision
    for b =1:size(confMat,1)
        precision(b) = confMat(b,b)/sum(confMat(:,b));
    end
    precision(isnan(precision))=[0];
    Precision = sum(precision)/size(confMat,1);
    PrecisionALL(i) = Precision;
    disp(Recall);
    
    % F-score
    F_score = 2*Recall*Precision/(Precision+Recall);
    F_scoreALL(i)= F_score;
    disp(F_score);
    
    %Check for Incorrectly classified images
    testImageLabels = testImages.Labels;
    Labels = cat(2, testImageLabels, predictedLabels);
    A = (testImageLabels == predictedLabels);
    [row,col,v] = find(A==0);
    
    missclassifiedALL{i} = v;
    disp(v);
    
    %Get the incorrectly classified Labels
    %Predicted Labels
    for j = 1:length(v)
        PL(j) = predictedLabels(row(j));
        [row1,col1,v1] = find(testImages.Labels == PL);
    end
    %True Test Labels
    for m = 1:length(v)
        TL(m) = testImageLabels(row(m));
    end
    
     PredLabel{i} = PL;
     TestLabel{i} = TL;
    
end

