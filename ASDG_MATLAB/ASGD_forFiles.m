% % for ff = 1:totalExpFiles

%     %Comment out this code%%%%%%%%%%%%%%%%%%%%%%%%%
%     if (ff ~= 1) % only for given data set(s) 1=> breast cancer, 7 => new-thyroid, 2=> cancer
%         continue;
%     end
%     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




    keepMainVars = {'dirExp', 'filesExp','totalExpFiles','ff', 'algos','iAlgo', 'algoFig', ...
        'bMNIST','bPlot','CPUs','gbX', 'gbY', 'gbDataBatch','ropeTeamSz', 'data', 'file_path'};
    clearvars('-except', keepMainVars{:});  


    algoFig = figure('Name',char([iAlgo ' Figure']));
    
    %%file_path
    %%size(data)
	
	filesExp(ff).name
    
% %     file_path = []; set it in Java considering multiple runs
% %	data = []; set it in Java considering multiple runs

	data = [];

    NC = 0; %Number of classes    
%     T = 0;
    Epochs = 0;

    if(~bMNIST)
        [NC, data, file_path, x, y, N, d, inputVal, givenOut, valData, valOut] = readData(data, file_path,filesExp(ff).name,filesExp(ff).folder);
		[~,file_used,~] = fileparts(file_path); %get only file name. Remove extension.
        if(NC == 2)
            NC = 1;
            activation = 'cross-entropy';
        elseif(NC>2)
            warning('Note: For multi-class classification, class labels MUST be from 0 to %d!\n',NC-1);
            activation='softmax';
        else
            warning('one class problem is not supported!\n');
            exit
        end
%         range = ceil(N*(fold-1)/fold); %for one fold. 
%         T = 1000;% 2*range*fold; %multiply by 2 is because of another "consistent run".
        Epochs = -1; % set by Java program50; %0.1
    else
        images = loadMNISTImages('data/train-images.idx3-ubyte');
        labels = loadMNISTLabels('data/train-labels.idx1-ubyte');
        x=images;
        [d,N] = size(x);
        newX0 = ones(1,N);
        x = vertcat(newX0, x);
        d = d+1;
        y = labels;
        y = y';
        images = loadMNISTImages('data/t10k-images.idx3-ubyte');
        labels = loadMNISTLabels('data/t10k-labels.idx1-ubyte');
        inputVal = images;
        [~,tmp] = size(inputVal);
        newX0 = ones(1,tmp);
        inputVal = vertcat(newX0, inputVal);
        givenOut = labels;
        givenOut = givenOut';
        file_path = 'data/train-images.idx3-ubyte';
        [~,file_used,~] = fileparts(file_path); %get only file name. Remove extension.
        NC = 10; %Number of classes
        activation='softmax';
%         T = 10000;
        Epochs = -1; %100 % now set by Java
    end

    gbX = x;
    gbY = y;

    eta = 0.1;
    NFCsgd = 0;
    W=zeros(d,NC);
    Whistory = cell(NC,2);
    gradHist = zeros(d, NC);
    r = zeros(d, NC);
    s = zeros(d, NC); 
    mAdam = zeros(d, NC);
    vAdam = zeros(d, NC);

    Wsgd = zeros(d,NC);
    WsgdHistory = cell(NC,2);
    gradHistSGD = zeros(d, NC);
    rSGD = zeros(d, NC);
    sSGD = zeros(d, NC);
    mAdamSGD = zeros(d, NC);
    vAdamSGD = zeros(d, NC);
%     Wmom = zeros(d,1); %momentum, currently working with Wsgd variable....
    Wprev = [];

    plotEsgd = [];
    plotEsgdGens = [];

    t = 0;
    ep = 0;
    idx = randperm(N);
    miniBatch = 10;

    plotNow = false;
% % end %ff file