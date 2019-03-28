%% Script to run Hamiltonian MC

%% Import statements
% addpath('mcmc/');
addpath('util/');
% addpath('/Users/ines/Documents/MATLAB/Euler');

%% Function to get the desired index directly off an output
% Copied from https://stackoverflow.com/questions/3627107/how-can-i-index-a-matlab-array-returned-by-a-function-without-first-assigning-it
paren = @(x, varargin) x(varargin{:});
curly = @(x, varargin) x{varargin{:}};

%% Define the parameters necessary for kinetic energy function (assumed gaussian)
n_dim = 2;
mu = repmat(0, 1, n_dim);
Sigma = repmat(1, 1, n_dim); % Sigma for each variable p_i, here considered to have 0 covariance

%% For the potential energy
% Assume you know the posterior up to the normalization constant and that
% it is a gaussian mixture
mu_tilde = [-10, 20];
sigma_tilde = [2, 7];
weights = [.5,.5];
p_tilde = @(x) gaussian_mix(x, mu_tilde, sigma_tilde, weights, 0);
U = @(q) -log(p_tilde(q));

% Cuidado com a função gradient!!
% FX = gradient(F) returns the one-dimensional numerical gradient of 
% vector F. The output FX corresponds to ∂F/∂x, which are the differences 
% in the x (horizontal) direction. The spacing between points is ASSUMED 
% to be 1.
spacing = 1
grad_U = @(q)subsref(gradient(U(q-3:spacing:q+3)), struct('type', '()', 'subs', {{ceil(length(q-1:spacing:q+1)/2)}}));

%% Other parameters for the HMC (leapfrog) algorithm
epsilon = 0.2; % Neal,  p. 141: " We must make epsilon proportional to d^{-1/4} to maintain a reasonable acceptance rate.
L = 10;
q0 = 0;

% Run the Hamiltonian algorithm
n_iter = 10^4;
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
samples_plotted = cell2mat(samples(1:10^1:end));
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

%% If you import results from the cluster and want to visualize them:
samples = load('samples_hmc_mat');
samples = samples.samples_hmc_mat;

figure(1)
x = -30:0.1:40;
plot(x, p_tilde(x)); hold on;
samples_plotted = samples(1:10^5:end);
c = linspace(1,10,length(samples_plotted)); % more yellow is further along the line
scatter(samples_plotted, zeros(1,length(samples_plotted)), [], c);
xlabel('x');
ylabel('p(x)');
title('Exploration of 1D Gaussian mixture by HMC')


figure(2)
histogram(samples, 'FaceColor', [100 149 237]/255, 'NumBins', 100);
title("Hamiltonian Metropolis Hastings");
xlabel('X');
ylabel('Absolute frequency');


