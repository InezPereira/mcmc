%% Script to run Hamiltonian MC

%% Import statements
addpath('../');
addpath('/Users/ines/Documents/MATLAB/Euler');

%% Function to get the desired index directly off an output
% Copied from https://stackoverflow.com/questions/3627107/how-can-i-index-a-matlab-array-returned-by-a-function-without-first-assigning-it
paren = @(x, varargin) x(varargin{:});
curly = @(x, varargin) x{varargin{:}};

%% Define the parameters necessary for kinetic energy function (assumed gaussian)
n_dim = 1;
mu = repmat(0, 1, n_dim);
Sigma = repmat(1, 1, n_dim); % Sigma for each variable p_i, here considered to have 0 covariance

%% For the potential energy
% Assume you know the posterior up to the normalization constant and that
% it is a gaussian mixture
mu_tilde = [-10, 20];
sigma_tilde = [5, 7];
weights = [1,1];
U = @(q)gaussian_mix(q, mu_tilde, sigma_tilde, weights, 0);
grad_U = @(q)subsref(gradient(U(q-1:0.1:q+1)), struct('type', '()', 'subs', {{ceil(length(q-1:0.1:q+1))}}));

m%% Other parameters for the HMC (leapfrog) algorithm
epsilon = 0.1;
L = 200;
q0 = 0;

% Run the Hamiltonian algorithm
n_iter = 10^5;
samples = {q0};
for ii=1:n_iter
    samples{ii+1} = hmc_neal(U, grad_U, mu, Sigma, epsilon, L, samples{ii});
end

figure(1)
histogram(cell2mat(samples), 'FaceColor', [100 149 237]/255, 'NumBins', 100);
title("Hamiltonian Metropolis Hastings");
xlabel('X')
ylabel('Absolute frequency');
savefig('Hybrid_MC')
