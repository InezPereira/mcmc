function p = gaussian_mix(x, list_mu, list_sigma, normalize)
    if normalize ~=1 && normalize ~= 0
        error('Not an allowed value for the normalize argument! Pick either 0 or 1!')
    end
    
    p = zeros(size(x));
    
    if size(list_mu) ~= size(list_sigma)
        error('List of mu and sigma values must have the same size!')
    else
        for i=1:length(list_mu)
        p = p + normpdf(x, list_mu(i), list_sigma(i));
        end
    end
    
    if normalize == 1
        p = p/length(list_mu);
%     elseif normalize == 0
%         disp('I am not normalizing, as you requested.')     
    end
end