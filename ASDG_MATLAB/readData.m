% x - training data input
% y - training data output
% N - size of training data set
% d - dimension of data set
% inputVal - testing data input
% givenOut - testing data output
function [NC, data, file_path, x, y, N, d, inputVal, givenOut, valData, valOut] = readData(data, file_path, givenFileName, givenFilePath)
%%%%%%%%% Training Data %%%%%%%%%%%%%%%%%%
data_type = '%g';
data_separator = ',';
file_type = '*.data';
   
if(isempty(data) || isempty(file_path)) %For first instance
    [data file_path] = get_data(data_type,data_separator,'Raw Data File',file_type, givenFileName, givenFilePath);   
end
% [~,file_used,~] = fileparts(file_path); %get only file name. Remove extension.

rawData = data(:,1:size(data,2)-1);
[N,d] = size(rawData);
for i = 1:d
    rawData(:,i) = norm_variance(rawData(:,i));
    rawData(:,i) = norm_scale01(rawData(:,i));
end
order = randperm(N);
tr = order(1:N*0.6);
ts = order(N*0.6+1:N);
x = rawData(tr,:);
newX0 = ones(size(x,1),1);
x = [newX0 x];
x = x';
[d, N] = size(x);
rawOut = data(:,size(data,2));
% % clear data;
y = rawOut(tr,:);
y = y';
NC = max(size(unique(y)));
% >>
%%%%%%%%%%%%%%% Training Data %%%%%%%%%%%%%%%%%%%%

%% Testing Data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%<<
inputVal = rawData(ts,:);
newX0 = ones(size(inputVal,1),1);
inputVal = [newX0 inputVal];
inputVal = inputVal';
[dt, Nt] = size(inputVal);

givenOut = rawOut(ts,:);
givenOut = givenOut';
%>>
%%%%%%%%%% test data %%%%%%%%%%%%%%%%%%%%%%

%% Validation Data
valData = inputVal(:,1:floor(Nt*0.5)); %50% of testing i.e. 20% val, 20% testing, 60% training
valOut = givenOut(1:floor(Nt*0.5));

%%inputVal = inputVal(:,floor(Nt*0.5)+1:end); %50% of testing i.e. 20% val, 20% testing, 60% training
%%givenOut = givenOut(floor(Nt*0.5)+1:end);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

end