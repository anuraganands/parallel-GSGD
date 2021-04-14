%% Initialize variables
% % function Initialize()
% %     clear all;
% %     clc;
% %     close all;
%     javarmpath([pwd '\ASJD_Java\dist\ASJD_Java.jar']);
%     
%     javaaddpath([pwd '\ASJD_Java\dist\ASJD_Java.jar']);
    
    algos = {'Canonical','Momentum','Nesterov','Adagrad','Adadelta','RMSprop','Adam'}; %{'Adagrad'}; %{'Canonical'}; %,
    bMNIST = false; %use or not to use MNIST datasets
    bPlot = true;
%     CPUs = 20; % for multi-threading
    
    if(isempty(data))
        if(bMNIST)
            filesExp = [];
            totalExpFiles = 1;
        else
            dirExp = uigetdir(pwd, 'Select a folder');
            filesExp = dir(fullfile(dirExp, '*.data'));
            totalExpFiles = size(filesExp,1);
        end
    end
    global gbX
    global gbY
    global gbDataBatch
    
    
% % end %Function
 