%% Ashynchronous SGD
function ASGD
    clear all;
    clc;
    close all;
    
    jarFile = 'C:\Users\sharma_au\Dropbox\ASGD\ASJD_Java\dist\ASJD_Java.jar';
    
    javarmpath(jarFile); %[pwd '\ASJD_Java\dist\ASJD_Java.jar']);
    javaaddpath(jarFile); %[pwd '\ASJD_Java\dist\ASJD_Java.jar']);
    
    algos = {'Canonical'}; %,{'Canonical','Momentum','Nesterov','Adagrad','Adadelta','RMSprop','Adam'}; %{'Adagrad'}; %
    bMNIST = false; %use or not to use MNIST datasets
    bPlot = true;
    CPUs = 20; % for multi-threading
    
    if(bMNIST)
        filesExp = [];
        totalExpFiles = 1;
    else
        dirExp = uigetdir(pwd, 'Select a folder');
        filesExp = dir(fullfile(dirExp, '*.data'));
        totalExpFiles = size(filesExp,1);
    end
    global gbX
    global gbY
    global gbDataBatch
    
    
for iAlgo = algos
    keepMainVars = {'dirExp', 'filesExp','totalExpFiles', 'algos','iAlgo', ...
        'bMNIST','bPlot', 'CPUs', 'gbX', 'gbY', 'gbDataBatch'};
    clearvars('-except', keepMainVars{:});
    
for ff = 1:totalExpFiles
    
    %Comment out this code%%%%%%%%%%%%%%%%%%%%%%%%%
    if (ff ~= 1) % only for given data set(s) 1=> breast cancer, 7 => new-thyroid, 2=> cancer
        continue;
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    
    
    
    keepMainVars = {'dirExp', 'filesExp','totalExpFiles','ff', 'algos','iAlgo', ...
        'bMNIST','bPlot','CPUs','gbX', 'gbY', 'gbDataBatch','ropeTeamSz'};
    clearvars('-except', keepMainVars{:});  
    
   
    file_path = [];
    data = [];
        
    NC = 0; %Number of classes    
%     T = 0;
    Epochs = 0;
    
    if(~bMNIST)
        [NC, data, file_path, x, y, N, d, inputVal, givenOut] = readData(data, file_path,filesExp(ff).name,filesExp(ff).folder);
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
        Epochs = 1000;
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
        Epochs = 10;
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
% %     fprintf('Iteration t = ');
%     while t < T 
    plotNow = false;
    while ep < Epochs
% %         fprintf('%d, ',t);
% %         if(mod(t,100) == 0)
% %             fprintf('\n');
% %         end

        gbDataBatch = cell(1,CPUs);

        for cpu = 1:CPUs
            if(mod(t,N)==0)
                idx = 1:N; %randperm(N);%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            end

            curIdx = idx(mod(t,N)+1:mod(t+miniBatch,N));
            if(isempty(curIdx))
                t = t+miniBatch;
                curIdx = (t-miniBatch)+1:N;
                curIdx = [curIdx 1:(t-curIdx)];
                t = -miniBatch;
                ep = ep+1;
                fprintf('ep: %d\n',ep);
                plotNow = true;
            end
            t = t+miniBatch;
            gbDataBatch{cpu} = curIdx;
        end
% %         jsonData = buildJason(dataBatch);
        

        %Now give it to slave threads
        
        
        % %     gr = getSGD(x, y, W, activation); %
        Jparam = cell(1,2);
%         Jparam{1} = gbX(:,curIdx);
%         Jparam{2} = gbY(curIdx);
        Jparam{1} = Wsgd;
        Jparam{2} = activation;
        jsonIn = buildJason(Jparam);

        
        
        JavaObj = asjd_java.RunInJava();
        val = JavaObj.startThreads(CPUs,'JgetSGD', jsonIn);
%         val = JavaObj.startThreads(10,'calcGrad', jsonIn);
        
        ret = '';
        while(isempty(ret))
            ret = JavaObj.getOutput();
            pause(0.001);
        end
% %         fprintf('ret: %s\n',ret);

        clear JavaObj

%         fprintf('ret: %s\n',ret);
        jsonOut = jsondecode(char(ret)); % jsonOut.Threads.ID0.json1

        for thrds = 1:CPUs
            str = ['jsonOut.Threads.ID'  num2str(thrds)  '.json1'];
            gri = eval(str);
            
            [Wsgd, sSGD, rSGD, gradHistSGD, WsgdHistory, mAdamSGD, vAdamSGD] = ...
            SGDvariation(gri, t, gbX(:,curIdx), gbY(curIdx), Wsgd, eta, activation, ...
            iAlgo, sSGD, rSGD, gradHistSGD, WsgdHistory, mAdamSGD, vAdamSGD);

        end

        
        NFCsgd = NFCsgd+1;
        
        if(plotNow) %mod(t,10) == 0)
            eSGD = 0;
            for i = 1:N
                eSGD = eSGD + getError(i,gbX,gbY,Wsgd,activation);
            end
            eSGD = eSGD/N;  
            plotEsgd = [plotEsgd eSGD]; 
            plotEsgdGens = [plotEsgdGens ep];
            
            plotNow = false;
        end
        
        
    end
    %SGD-canonical
    %<<

    %>>
    
    if(bPlot)
        hold off;
        
%         plot(tGSGDplot,GSGDplot,'-', 'color', clr); %tGSGDplot
%         hold on; 
        plot(plotEsgdGens,plotEsgd,'--','color', 'm');

        drawnow

        % range = sprintf('range*%d',range);
        xlabel('Selected Iterations','fontsize',10,'color','b')
        ylabel('Error (E_i_n)','fontsize',10,'color','b')
        title({['Performance: ASGD-' cell2mat(iAlgo)], file_used});
%         legend('GSGD','SGD');%,'Validation');
    end
end %ff file
end %iAlgo
end %Function
 

%% get error
% for SGD idx is 1:N but for GSGD its random
function e = getError(idx,x,y,W,type)
    e = 0;
    if(strcmp(type,'cross-entropy'))
        for in = [idx] %take the full training sequence/data of this fold   
            e = e + log(1+exp(-y(in)*W'*x(:,in)));   
        end                  
    elseif(strcmp(type,'softmax')) % W has d dimenstion (same as input) and NC possible outputs
        datasize = size(idx,2);        
        [d, NC] = size(W);
        labels = full(sparse([ 1:datasize+1],[y(idx)'+1;NC],1)); %groundTruth
        labels(end,:) = []; % labels (datasize x NC)
        
        lb = 1;
        for in = [idx] 
            logits = W'*x(:,in); % W'(NCxd) * x(dx1) = logits(NCx1); %intermediate output is known as logits in machine learning
            D = -max(logits); %to cater large numbers                       
        
            tmp = exp(logits+D)/sum(exp(logits+D));
            tmp = labels(lb,:)*tmp;
            lb = lb+1;
            tmp = 1/tmp;
            e = e + log(tmp);
        end
    else
        error('Error! Activation function not supported.\n');
        exit;
    end
    
    sz = max(size(idx));
    e = e/sz; %average error         
end


%% SGD gradient calculation
% cpuNo is integer
% jsonIn is a json String
function jsonOut = JgetSGD(jsonIn, varargin)
    jsonWrapper = jsondecode(jsonIn);
    
    global gbDataBatch
    global gbX
    global gbY
    cpuNo = 1;
    if(nargin == 2)
        cpuNo = varargin{1};
    end
    
% %     fprintf('cpu: %d\n',cpuNo);
% %     printVector(gbDataBatch{cpuNo});
    
    %<< (1) put function specific parameters here
%     xi = jsonWrapper.json1; %very heavy. Don't do it.
%     yi = jsonWrapper.json2;
    xi = gbX(:,gbDataBatch{cpuNo});
    yi = gbY(gbDataBatch{cpuNo});
    W = jsonWrapper.json1;
    type = jsonWrapper.json2;
    %>>
    
    %<< (2) call the corresponding Matlab function
    gr = getSGD(xi, yi, W, type);
    %>>
   
    %<< (3) now prepare the output
    out = cell(1,1);
    out{1} = gr;
    %>>
    
    jsonOut = buildJason(out); %x1
end

%% SGD gradient calculation
function gr = getSGD(xi, yi, W, type)
    [d, miniSz] = size(xi);
    
    gr = zeros(d,1); 
    for i = 1:miniSz
        if(strcmp(type,'cross-entropy'))
            s = W'*xi(:,i);
            gri = - yi(i)*xi(:,i)/(1+exp(yi(i)*s)); %W'*xi)); % W at t i.e. W(t)
        elseif(strcmp(type,'softmax'))
            s = W'*xi(:,i); % W'(NCxd) * x(dx1) = s(NCx1); 
            [d, NC] = size(W);
            D = -max(s);
            s = exp(s+D)/sum(exp(s+D)); % s(NCx1)

            datasize = size(xi(:,i),2);      
            labels = full(sparse([ 1:datasize+1],[yi(i)'+1;NC],1)); %groundTruth
            labels(end,:) = []; % labels (datasize x NC)
            labels = labels';
            try
                gri = (s-labels)*xi(:,i)';
                gri = gri';
            catch ME
                d
                NC
                fprintf('s: [%d %d]\n',size(s));
                fprintf('xi: [%d %d]\n',size(xi(:,i)));  
            end
        else
            error('Error! Activation function not supported.\n');
            exit
        end
        gr = gr + gri;
    end
    gr = (1/miniSz)*gr;
end

%% BGD gradient calculation
function gr = getBGD(N, d, x, y, W)
    gr = zeros(d,1);
    for i = 1:N
        gri = -y(i)*x(:,i)/(1+exp(y(i)*W'*x(:,i))); % W at t i.e. W(t)
        gr = gr + gri;
    end 
    gr = (1/N)*gr;
end


%% all variations of the SGD algorithm
function [W, s, r, gradHist, Whistory, m, v] = SGDvariation(gr, t, x, y, W, eta, activation, ...
    algo, s, r, gradHist, Whistory, m, v)
    beta = 0.9;
    epsilon = 1e-8;

    % gradient is computed Now in Parallel by slave CPUs
% %     gr = getSGD(x, y, W, activation); %

    if(strcmp(algo,'Canonical'))
        W = W-eta*gr;
    elseif(strcmp(algo,'Momentum'))
        bapplyMomentum = false;
        if(isempty(Whistory{1}) && isempty(Whistory{2})) 
            Whistory{1} = W;
        elseif(isempty(Whistory{2}))
            Whistory{2} = W;
        else % 2
            Whistory{1} = Whistory{2}; %Wt-2
            Whistory{2} = W; %Wt-1
            bapplyMomentum = true;
        end
        
        if(bapplyMomentum)
            rho = 0.9;
            W = Whistory{2}-eta*gr - rho*(Whistory{2}-Whistory{1});%momentum SGD
        else
            W = W-eta*gr;%simple SGD
        end
    elseif(strcmp(algo,'Nesterov'))
        error('Nesterov is Not supported for Parallel ASGD.\n');
        bapplyNesterov = false;
        if(isempty(Whistory{1}) && isempty(Whistory{2})) 
            Whistory{1} = W;
        elseif(isempty(Whistory{2}))
            Whistory{2} = W;
        else % 2
            Whistory{1} = Whistory{2}; %Wt-2
            Whistory{2} = W; %Wt-1
            bapplyNesterov = true;
        end
        
        if(bapplyNesterov)
            rho = 0.9;
            gr = getSGD(x, y, W-rho*Whistory{2}, activation);
            W = Whistory{2}-eta*gr - rho*(Whistory{2}-Whistory{1});%nesterov SGD
        else
            gr = getSGD(x, y, W, activation);
            W = W-eta*gr;%simple SGD
        end
    elseif(strcmp(algo,'Adagrad'))
        gradHist = gradHist + gr.^2;
        W = W - eta * gr ./ (sqrt(gradHist) + epsilon); 
        
    elseif(strcmp(algo,'Adadelta'))
        if(t==1)
            r = gr.^2;
        else
            r = beta * r + (1-beta)* gr.^2;  
        end  
        
        v1 = - (sqrt(s + epsilon)./sqrt(r + epsilon)) .* gr;
        % update accumulated updates (deltas)
        %out
        s = beta * s + (1-beta)* v1.^2;
        %out
        W = W + v1; 

    elseif(strcmp(algo,'RMSprop'))
        if(t==1)
            r = gr.^2;
        else
            r = beta * r + (1-beta)* gr.^2;  
        end
        W = W - eta * gr ./ sqrt(r + epsilon); 
    elseif(strcmp(algo,'Adam'))
        beta1 = 0.9;
        beta2 = 0.999;
        
        m = beta1.*m + (1 - beta1).*gr;
        % Update biased 2nd raw moment estimate
        v = beta2.*v + (1 - beta2).*(gr.^2);

        % Compute bias-corrected 1st moment estimate
        mHat = m./(1 - beta1^t);
        % Compute bias-corrected 2nd raw moment estimate
        vHat = v./(1 - beta2^t);

        % Update decision variables
        W = W - eta.*mHat./(sqrt(vHat) + epsilon);
    else %error
        W = [];
        s = [];
        gradHist = [];
        error('Incorrect choice of algorithm in SGDvariation!\n');
        exit
    end
    
            

end
