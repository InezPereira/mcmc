function final_visualizations(chain, samples)
% Figure to get the exploration space
figure(3)
contour(X,Y,Z), hold on
samples_plotted = cell2mat(samples{chain}');
samples_plotted = samples_plotted(1:10^2:end,:);
c = linspace(1,10,length(samples_plotted)); % more yellow is further along the line
% c2 = linspace(1,10,length(samples_plotted));
% c3 = linspace(1,10,length(samples_plotted));
% c = [c1; c2; c3]'
scatter(samples_plotted(:,1), samples_plotted(:,2), [], c);
line(samples_plotted(:,1), samples_plotted(:,2)) %, 'Color', c);
% plot(samples_plotted(:,1), samples_plotted(:,2), '-o', 'MarkerEdge', 'r');
title('Exploration of 2D Gaussian mixture by HMC')
hold off;


figure(4)
hist3(cell2mat(samples{chain}'), 'FaceColor', [100 149 237]/255, 'Nbins',[20,20]);
title("Hamiltonian Monte Carlo");
xlabel('X')
ylabel('Y')
zlabel('Absolute frequency');
% savefig('Hybrid_MC')

figure(5)
traceplots(nChains, samples, 1)