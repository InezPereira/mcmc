function p = gaussian_mix(x, list_mu, list_sigma, weights, normalize)
    if normalize ~=1 && normalize ~= 0
        error('Not an allowed value for the normalize argument! Pick either 0 or 1!')
    end
    
    p = zeros(size(x));
    
    if length(list_mu) ~= length(list_sigma) || length(list_mu) ~= length(weights)
        error('List of mu, sigma and weight values must have the same size!')
    else
        for ii=1:length(list_mu)
        p = p + weights(ii)*normpdf(x, list_mu(ii), list_sigma(ii));
        end
    end
    
    if normalize == 1
        p = p/length(list_mu);
%     elseif normalize == 0
%         disp('I am not normalizing, as you requested.')     
    end
end