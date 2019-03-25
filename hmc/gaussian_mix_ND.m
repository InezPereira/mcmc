function p = gaussian_mix_ND(X, mu_cell_array, sigma_cell_array, weights)
% Assumes first dimension of the X matrix is going to be the number of z's

% https://brilliant.org/wiki/gaussian-mixture-model/
% You need to check that the weights sum up to one!
if sum(weights)~=1
    error('Weights need to sum up to 1!');
else 
    
    p = zeros(length(X),1);
    
    if length(mu_cell_array) ~= length(sigma_cell_array) || length(mu_cell_array) ~= length(weights)
        error('List of mu, sigma and weight values must have the same length!')
    else
        for ii=1:length(mu_cell_array)
        p = p + weights(ii)*mvnpdf(X, mu_cell_array{ii}, sigma_cell_array{ii});
        end
    end
end
end