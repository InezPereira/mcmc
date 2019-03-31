%% Script to run Hamiltonian MC in 2D

%% Import statements
addpath('util/');
addpath('hmc/');

%% Define necessary parameters
[mu, Sigma, weights, p_tilde, U, grad_U] = define_param()

%% Initial plots of the distribution we want to approximate
first_visualizations(p_tilde)

%% Other parameters for the HMC (leapfrog) algorithm
epsilon = 0.2; % Neal,  p. 141: " We must make epsilon proportional to d^{-1/4} to maintain a reasonable acceptance rate.
L = 10;
q0 = [0,0];

% Run the Hamiltonian algorithm
n_iter = 10^3;
samples = {q0};
reject = 0;
for ii=1:n_iter
    [samples{ii+1}, reject] = hmc(U, grad_U, mu, Sigma, epsilon, 0, L, 0, samples{ii}, reject);
end

% samples_hmc_mat = cell2mat(samples)
% save('samples_hmc_mat','samples_hmc_mat')

%% Final visualizations

% Figure to get the exploration space
figure(3)
contour(X,Y,Z), hold on
samples_plotted = cell2mat(samples')
samples_plotted = samples_plotted(1:10^1:end,:)
c = linspace(1,10,length(samples_plotted)); % more yellow is further along the line
% c2 = linspace(1,10,length(samples_plotted));
% c3 = linspace(1,10,length(samples_plotted));
% c = [c1; c2; c3]'
scatter(samples_plotted(:,1), samples_plotted(:,2), [], c);
line(samples_plotted(:,1), samples_plotted(:,2)) %, 'Color', c);
% plot(samples_plotted(:,1), samples_plotted(:,2), '-o', 'MarkerEdge', 'r');
title('Exploration of 2D Gaussian mixture by HMC')


figure(4)
hist3(cell2mat(samples'), 'FaceColor', [100 149 237]/255, 'Nbins',[20,20]);
title("Hamiltonian Metropolis Hastings");
xlabel('X')
ylabel('Y')
zlabel('Absolute frequency');
savefig('Hybrid_MC')