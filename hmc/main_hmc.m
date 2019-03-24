%% Script to run Hamiltonian MC

%% Import statements
addpath('../');
% addpath('/Users/ines/Documents/MATLAB/Euler');

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
sigma_tilde = [2, 7];
weights = [1,1];
p_tilde = @(x) gaussian_mix(x, mu_tilde, sigma_tilde, weights, 0);
U = @(q) -log(p_tilde(q));
grad_U = @(q)subsref(gradient(U(q-1:0.1:q+1)), struct('type', '()', 'subs', {{ceil(length(q-1:0.1:q+1))}}));

%% Other parameters for the HMC (leapfrog) algorithm
epsilon = 0.1;
L = 10;
q0 = 0;

% Run the Hamiltonian algorithm
n_iter = 10^2;
samples = {q0};
reject = 0;
for ii=1:n_iter
    [samples{ii+1}, reject] = hmc_neal(U, grad_U, mu, Sigma, epsilon, L, samples{ii}, reject);
     
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
samples_plotted = cell2mat(samples(1:10^3:end))
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

%% Stuff



