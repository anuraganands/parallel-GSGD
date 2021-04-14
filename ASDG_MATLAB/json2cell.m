%% input json string (encoded)
% returns decoded cell
function retCell = json2cell(json)
    Level1 = jsondecode(json);
    
    r = 1;
    try
        while true
           str = ['Level1.row' num2str(r)];  
           Level2{r} = eval(str);
           r = r+1;
        end
    catch E
        rows = r-1; %total rows
    end
    
    for r = 1:rows
        c = 1;
        try
            while true
               str = ['Level2{' num2str(r) '}' '.json' num2str(c) ];  
               retCell{r,c} = eval(str);
               c = c+1;
            end
        catch E
%             cols = c-1 %total rows
            continue
        end
    end

end