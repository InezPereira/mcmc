% https://stats.stackexchange.com/questions/507/what-is-the-best-method-for-checking-convergence-in-mcmc
% https://stats.stackexchange.com/questions/4258/can-i-semi-automate-mcmc-convergence-diagnostics-to-set-the-burn-in-length

%% Traceplots

function traceplots(nChain, samples, dim)
% nChain = 3;
% dim = 1;
colors = generate_colors(nChains);
% colors = ['r','g','b'];
for ii=1:nChain
    samples_plotted = cell2mat(samples{ii}')
    samples_plotted = samples_plotted(:,dim)
%     plot(cell2mat(samples{ii}'), 'color', cell2mat(colors(ii)));
    plot(samples_plotted, 'color', cell2mat(colors(ii)));
    hold on;
    title('Traceplot')
    xlabel('Iteration')
    ylabel('Sample value')
end