% How to create the proposal distribution
% Assuming 1D distributions:
% Iterations:
% Proposal distribution: 
%   - one has previous state as mean and all have the
% same sigma (alternatives possible). 
%   - The other distributions will convey different means. We can set 
% as means the modes of p_tilde.

function samples = mh_random_walk_mixture_proposals(n_iter, initial_state, Sigma, weights_q, weights_p, mu_tilde, sigma_tilde)
    samples = {initial_state};
    for ii=1:n_iter
        % Define x = x^S
        x = samples{ii};
        % Sample x' ~ q(x'|x)
        x_prime = normrnd(x,Sigma);
        % Compute acceptance probability alpha
        p_tilde_x_prime = gaussian_mix(x_prime, mu_tilde, [1,1], sigma_tilde, weights_p);
        q_x_given_x_prime = gaussian_mix(x, [x_prime, mu_tilde], [1, sigma_tilde], weights_q);
        p_tilde_x = gaussian_mix(x, mu_tilde, [1,1], sigma_tilde, weights_p);
        q_x_prime_given_x = gaussian_mix(x_prime, [x, mu_tilde], [1, sigma_tilde], weights_q);
        
        alpha = (p_tilde_x_prime*q_x_given_x_prime)/(p_tilde_x*q_x_prime_given_x);
        % Compute r = min(1,alpha)
        r = min(1,alpha);
        % Sample u ~ U(0,1)
        u = rand();
        % Set new sample to x^S+1 = x' if u<r OR x^S+1 =x^S if u>=r
        if u<r
            samples{ii+1} = x_prime;
        else
             samples{ii+1} = x;
        end
    end

disp('Done with my Metropolis Hastings random walk!')
end


