%%
% para should be col-wise cell i.e. 1xN.
function encodedJason = buildJason(para)
    if(~iscell(para))
        error('para is not a cell\n');
    end

    jsonStr = '{';
    for i = 1:size(para,2)
         variJson = jsonencode(para{i});
         jsonStr = [jsonStr '"json' num2str(i) '":' variJson ','];
    end
    jsonStr = jsonStr(1:end-1);
    jsonStr = [jsonStr '}'];
    
    encodedJason = jsonStr;
end