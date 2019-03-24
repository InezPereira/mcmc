%% Function to generate one sample from a gaussian conditional distribution
% Define the conditional distributions
% Based on: https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Conditional_distributions
% Discrepancy between Wiki and http://fourier.eng.hmc.edu/e161/lectures/gaussianprocess/node7.html

% The fact that there is symmetry in the formula is proven here: http://cs229.stanford.edu/section/more_on_gaussians.pdf

function sample = gaussian_conditional_sample(mu, Sigma, sample_idx, condition_idx, observ)
mu_bar = mu(sample_idx) + Sigma(sample_idx,condition_idx)*inv(Sigma(condition_idx,condition_idx))*(observ-mu(condition_idx));
Sigma_bar = Sigma(sample_idx,sample_idx) - Sigma(sample_idx,condition_idx)*Sigma(condition_idx,condition_idx)^(-1)*Sigma(condition_idx,sample_idx);
% Generating sample from this distribution
sample = normrnd(mu_bar,Sigma_bar)
end