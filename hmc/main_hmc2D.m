%% Script to run Hamiltonian MC in 2D

%% Import statements
% addpath('mcmc/');
addpath('util/');
% addpath('/Users/ines/Documents/MATLAB/Euler');

%% Define the parameters necessary for kinetic energy function (assumed gaussian)
n_dim = 2;
mu = repmat(0, 1, n_dim);
Sigma = repmat(1, 1, n_dim); % Sigma for each variable p_i, here considered to have 0 covariance

%% For the potential energy
% Assume you know the posterior up to the normalization constant and that
% it is a gaussian mixture
mu_tilde = {[-5, -5], [0,0], [3,3]};
sigma_tilde = {[2, 0; 0,2], [5,0;0,5], [4,0;0,4]};
weights = [1/3,1/3,1/3];
p_tilde = @(X) gaussian_mix_ND(X, mu_tilde, sigma_tilde, weights);
U = @(q) -log(p_tilde(q));
grad_U = @(q, U)gradient_2D(q,U);

% grad_U = @(q)
% 
% spacing = 1
% grad_U = @(q)subsref(gradient(U(q-8:spacing:q+8)), struct('type', '()', 'subs', {{ceil(length(q-1:0.1:q+1))}}));

%% Stackoverflow gradient calculation

nd = sum(size(F)>1);


%% Initial plots of the distribution we want to approximate
x = -15:.1:10; %// x axis
y = -15:.1:10; %// y axis

[X, Y] = meshgrid(x,y);
input = [X(:) Y(:)];
Z = p_tilde(input);
Z = reshape(Z,size(X)); %// put into same size as X, Y

figure(1)
surf(X,Y,Z) %// 3D plot

figure(2)
contour(X,Y,Z), axis equal  %// contour plot; set same scale for x and y...

%% Other parameters for the HMC (leapfrog) algorithm
epsilon = 0.2; % Neal,  p. 141: " We must make epsilon proportional to d^{-1/4} to maintain a reasonable acceptance rate.
L = 10;
q0 = [0,0];

% Run the Hamiltonian algorithm
n_iter = 10^3;
samples = {q0};
reject = 0;
for ii=1:n_iter
    [samples{ii+1}, reject] = hmc_neal(U, grad_U, mu, Sigma, epsilon, 0, L, 0, samples{ii}, reject);
    
%     % Diagnostic plot:
%     if mod(ii, 5)==0
%      H = samples{ii+1}*inv(sigma_tilde)*samples{ii+1}'
%      figure(1.1)
%      plot(ii, H); hold on
%      xlabel('Iteration')
%      ylabel('Hamiltonian')
%     end
end

% samples_hmc_mat = cell2mat(samples)
% save('samples_hmc_mat','samples_hmc_mat')

%% Final visualizations

% Figure to get the exploration space
figure(1)
x = -30:0.1:40;
plot(x, p_tilde(x)); hold on
samples_plotted = cell2mat(samples(1:10^1:end))
c = linspace(1,10,length(samples_plotted)); % more yellow is further along the line
scatter(samples_plotted, zeros(1,length(samples_plotted)), [], c);
xlabel('x');
ylabel('p(x)');
title('Exploration of 1D Gaussian mixture by HMC')


figure(2)
histogram(cell2mat(samples), 'FaceColor', [100 149 237]/255, 'NumBins', 100);
title("Hamiltonian Metropolis Hastings");
xlabel('X')
ylabel('Absolute frequency');
savefig('Hybrid_MC')