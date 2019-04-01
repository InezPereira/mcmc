function p = gaussian_mix(x, list_mu, list_sigma, weights)
    % https://brilliant.org/wiki/gaussian-mixture-model/
    % You need to check that the weights sum up to one!
    if sum(weights)~=1
        error('Weights need to sum up to 1!');
    else 
        p = zeros(size(x));

        if length(list_mu) ~= length(list_sigma) || length(list_mu) ~= length(weights)
            error('List of mu, sigma and weight values must have the same size!')
        else
            for ii=1:length(list_mu)
            p = p + weights(ii)*normpdf(x, list_mu(ii), list_sigma(ii));
            end
        end
    end
end