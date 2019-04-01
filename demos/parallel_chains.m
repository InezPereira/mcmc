%% Demo script to run Hamiltonian MC in 2D

%% Import statements
addpath('../');
addpath('util/');
addpath('algorithms/hmc/');
addpath('diagnostics/');

%% Define necessary parameters
n_dim = 2
[mu, Sigma, weights, p_tilde, U, grad_U] = define_param(n_dim);

%% Initial plots of the distribution we want to approximate
first_visualizations(p_tilde);

%% Other parameters for the HMC (leapfrog) algorithm
epsilon = 0.1; % Neal,  p. 141: " We must make epsilon proportional to d^{-1/4} to maintain a reasonable acceptance rate.
L = 20;
q0 = {[-20,-10], [0,0], [40,0]}
temp = repmat(1.1, 1, n_dim)
nChains = 3;
samples = {};

tic
% ticBytes(gcp);
for ii = 1:nChains
% Run the Hamiltonian algorithm
    n_iter = 10^4;
    samples{ii}= {cell2mat(q0(ii))};
    reject = 0;
    for jj=1:n_iter
        [samples{ii}{jj+1}, reject] = hmc(U, grad_U, mu, Sigma, epsilon, 1, L, 1, samples{ii}{jj}, reject);

        if mod(jj,1000) ==0
            reject_rate = reject/jj;
            fprintf('Performing iteration number: (%d)\n', jj)
            fprintf('Current rejection rate: (%d)\n', reject_rate)
        end
        
        if jj == n_iter
            final_reject(ii) = reject / n_iter;
        end
    end
end
% tocBytes(gcp)
toc

save('samples','samples')

%% Final visualizations
chain = 2 % Choose a chain
final_visualizations(chain)