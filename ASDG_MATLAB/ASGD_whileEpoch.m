% while ep < Epochs

%     gbDataBatch = cell(1,CPUs);
% 
%     for cpu = 1:CPUs
%         if(mod(t,N)==0)
%             idx = 1:N; %randperm(N);%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%         end
% 
%         curIdx = idx(mod(t,N)+1:mod(t+miniBatch,N));
%         if(isempty(curIdx))
%             t = t+miniBatch;
%             curIdx = (t-miniBatch)+1:N;
%             curIdx = [curIdx 1:(t-curIdx)];
%             t = -miniBatch;
%             ep = ep+1;
%             fprintf('ep: %d\n',ep);
%             plotNow = true;
%         end
%         t = t+miniBatch;
%         gbDataBatch{cpu} = curIdx;
%     end
        
% get gri
    
% Update parameter

    NFCsgd = NFCsgd+1;

    if(plotNow) %mod(t,10) == 0)
        eSGD = 0;
        for i = 1:N
            eSGD = eSGD + getError(gbX(:,i),gbY(i),Wsgd,activation);
        end
        eSGD = eSGD/N;  
        plotEsgd = [plotEsgd eSGD]; 
        plotEsgdGens = [plotEsgdGens ep];

        plotNow = false;
    end
% end