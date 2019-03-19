%% Metropolis Hastings algorithm

%% Standard import statements
addpath('../');

%% Unnormalized function whose normalized version you want to approximate
x = [-30:0.1:40]
mu_tilde = [-10, 20];
sigma_tilde = [5, 7];
p_tilde = gaussian_mix(x, mu_tilde, sigma_tilde, [1,1], 0);

figure(1)
plot(x, p_tilde)
title('p_{tilde}')
xlabel('X')
ylabel('Density')

%% Proposal distribution
% ﻿The user is free to use any kind of proposal they want, subject to
% some conditions.
% A commonly used proposal is a symmetric Gaussian distribution centered 
% on the current state, q(x'|x)= N(x'|x,Σ); this is called a random walk 
% Metropolis algorithm.
% ﻿This is a somewhat tricky target distribution, since it consists of 
% two well separated modes. It is very important to set the variance of the 
% proposal v correctly: If the variance is too low, the chain will only 
% explore one of the modes, as shown in Figure 24.7(a), but if the variance 
% is too large, most of the moves will be rejected, and the chain will be 
% very sticky, i.e., it will stay in the same state for a long time. This

%% Initialize x0 and define number of iterations

x0 = normrnd(0,1)

%% For loop for MH random walk

n_iter = 10^6;
x_list = {x0};
Sigma = 1; % How to choose the variance??

samples = mh_random_walk(n_iter, x0, Sigma, [1,1], mu_tilde, sigma_tilde)

%% Plotting

samples_mat = cell2mat(samples)

figure(2)
histogram(samples_mat, 'FaceColor', [254,178,76]/255, 'NumBins', 100);
title("Metropolis Hastings random walk - Initialization as N(0,1)");
xlabel('X')
ylabel('Absolute frequency');
% axis([-10 30 0 Inf]);

% With a variance of 1, you indeed only explore one of the modes!
% It probably also depends on the initialization.

%% MH random walk with mixture proposals

x0 = 20
n_iter = 10^7;
x_list = {x0};
Sigma = 6   ; % How to choose the variance??

weights = [1/3, 1/3, 1/3];
samples_mixture_prop = mh_random_walk_mixture_proposals(n_iter, x0, Sigma, weights, mu_tilde, sigma_tilde)

samples_mixt_mat = cell2mat(samples_mixture_prop)

figure(3)
histogram(samples_mixt_mat, 'FaceColor', [254,178,76]/255, 'NumBins', 100);
title("Metropolis Hastings random walk with mixture proposals");
xlabel('X')
ylabel('Absolute frequency');

%% Hybrid/Hamiltonian Monte Carlo

n_iter = 10^3
L = 10
rho = 1
mu_tilde = [-10, 20];
sigma_tilde = [5, 7];
x0 = -10
samples_hybrid = hybrid_mc(n_iter, L, x0, rho, mu_tilde, sigma_tilde)

samples_hybrid_mat = cell2mat(samples_hybrid)

figure(4)
histogram(samples_hybrid_mat, 'FaceColor', [254,178,76]/255, 'NumBins', 100);
title("Hamiltonian Metropolis Hastings");
xlabel('X')
ylabel('Absolute frequency');