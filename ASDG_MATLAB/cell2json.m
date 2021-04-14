%%
% para should be col-wise cell i.e. rows x N.
function encodedJason = cell2json(para)
    if(~iscell(para))
        error('para is not a cell\n');
    end

    jsonStr = '{';
    for r = 1:size(para,1)
        jsonStr = [jsonStr '"row' num2str(r) '":{'];
        for i = 1:size(para,2)
             variJson = jsonencode(para{r, i});
             jsonStr = [jsonStr '"json' num2str(i) '":' variJson ','];
        end
        jsonStr = jsonStr(1:end-1); %remove last comma
        jsonStr = [jsonStr '},'];
    end
    jsonStr = jsonStr(1:end-1);
    jsonStr = [jsonStr '}'];
    
    encodedJason = jsonStr;
end