resFile = ['Result/' cell2mat(iAlgo) '/' file_used '_accuracy.csv' ];

results = [CPU DELAY_TOLERANCE Guided MINI_BATCH Epochs Eout NFC];

if exist(resFile, 'file') ~= 2 %does not exist
    fid=fopen(resFile,'wt');
    x={'CPU' 'DELAY_TOLERANCE' 'Guided' 'MINI_BATCH' 'Epochs' 'Eout' 'NFC'};
    [rows,cols]=size(x);
    for i=1:rows
          fprintf(fid,'%s,',x{i,1:end-1});
          fprintf(fid,'%s\n',x{i,end});
    end
    fclose(fid);
end  

dlmwrite(resFile,results,'-append', 'precision','%.3f');