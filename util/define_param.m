function [mu, Sigma, weights, p_tilde, U, grad_U] = define_param(n_dim)

    %% Define the parameters necessary for kinetic energy function (assumed gaussian)
    mu = repmat(0, 1, n_dim);
    Sigma = repmat(1, 1, n_dim); % Sigma for each variable p_i, here considered to have 0 covariance

    %% For the potential energy
    % Assume you know the posterior up to the normalization constant and that
    % it is a gaussian mixture
    % mu_tilde = {[-5, -5], [0,0], [3,3]};
    n_mix = 3
    mu_tilde = {repmat(-5, 1, n_dim), repmat(0, 1, n_dim), repmat(3, 1, n_dim)}
    sigma_tilde = {2*eye(n_dim), 3*eye(n_dim), 4*eye(n_dim)}; % Assumption of independence made (no covariance)
    weights = repmat(1/n_mix, 1, n_mix);
    p_tilde = @(X) gaussian_mix(X, mu_tilde, sigma_tilde, weights);
    U = @(q) sum(-log(p_tilde(q)));
    grad_U = @(q, U)gradient_ND(q,U);

end