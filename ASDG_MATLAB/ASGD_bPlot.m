if(bPlot)
    hold off;

%         plot(tGSGDplot,GSGDplot,'-', 'color', clr); %tGSGDplot
%         hold on; 
    algoFig = figure(algoFig);
    
    plot(plotEsgdGens,plotEsgd,'--','color', 'm');

    drawnow

    % range = sprintf('range*%d',range);
    xlabel('Selected Iterations','fontsize',10,'color','b')
    ylabel('Error (E_i_n)','fontsize',10,'color','b')
    title({['Performance: ASGD-' cell2mat(iAlgo)], file_used});
%         legend('GSGD','SGD');%,'Validation');

	outFile = sprintf('%s', outFile);
    %outFile = ['Result/' cell2mat(iAlgo) '/' file_used '_trial.fig' ];
    savefig(algoFig, outFile);
    close(algoFig);

end