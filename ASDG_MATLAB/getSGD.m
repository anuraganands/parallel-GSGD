
%% SGD gradient calculation
function gr = getSGD(xi, yi, W, type)
    [d, miniSz] = size(xi);
    if(isrow(W))
        W = W';
    end
    NC = size(W,2);
    gr = zeros(d,NC);
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
