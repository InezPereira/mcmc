%% Function to implement hybrid/hamiltonian Monte Carlo
% Hybrid/hamiltonian MCMC ﻿is an MCMC algorithm that incorporates 
% information about the gradient of the target distribution to improve 
% mixing in high dimensions.

% Implementation follows the notation in the paper "An Introduction to MCMC
% for Machine Learning", by Andrieu et al, 2003.
% It was really super helpful to understand how hybrid MC works!

% Issues: 
% - When computing gradients, should I be doing it with respect to p_tilde
% (aka the unnormalized posterior)?
% - May need to adapt the dimensions of the the gaussian from which we
% sample u_prime

function samples = hybrid_mc(n_iter, L, rho, p_tilde, mu_tilde. sigma_tilde)
% Creating necessary data structures
x = {}
x_subloop = {}
u = {}
% For i = 1:N
    for ii=1:n_iter
    % Sample v ~ U(0,1) and u^* ~ N(0,I_n)
    v = rand();
    u_prime = normrnd(0,1);
    % Let x0 = x(i) and u0 = u^* + rho*∆(x0)/2
    x0 = x{ii};
    grad = gradient(log(p_tilde)) % you first compute the numerical gradient for the whole of p_tilde
    u0 = u_prime + rho*grad(find(x==x0))/2
    % For l = 1:L take steps
        for l = 2:L % Because indexing in Matlab starts at 1 and not zero :)
            % x_l = x_{l-1} + rho*u_{l-1}
            x_subloop{1} = x{ii}
            x_subloop{l} = x{l-1} + rho*u{l-1}
            % u_l = u_{l-1} + rho*∆(x_l)
            u{l} = u{l-1} + rho*grad(find(x==x{l}))/2
            % where rho_l = rho for l<L and rho_L = rho/2
        end
        p_tilde_xL = gaussian_mix(x_subloop{L}, mu_tilde, sigma_tilde, ones(1,length(mu_tilde)), 0);
        p_tilde_xii = gaussian_mix(x{ii}, mu_tilde, sigma_tilde, ones(1,length(mu_tilde)), 0);
        alpha = min(1, p_tilde_xL/p_tilde_xii*exp(-1/2*(u{L}'*u{L}-u_prime'*u_prime)))

        % If v < A = min(1, p(x_L)/p(x^i)exp(-1/2*(u_L^T*u_L-u^*Tu^*)))
            % (x^{i+1}, u^{i+1}) = (x_L, u_L)
        if v < alpha
            x{ii+1} = x_subloop{L}
            u{ii+1} = u{L}
        else
            x{ii+1} = x{ii}
            u{ii+1} = u_prime
            % Else: (x^{i+1}, u^{i+1}) = (x^i, u^*)
        % Marginal samples from p(x) are ob- tained by simply ignoring u.
        end
    end
    
    samples = x
end