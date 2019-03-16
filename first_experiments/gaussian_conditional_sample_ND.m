%% Function to generate one sample from a gaussian conditional distribution
% Generalization of the bivariate case
% https://math.stackexchange.com/questions/3145228/conditional-multivariate-normal-beyond-the-bivariate-case
% Great solution here: https://stats.stackexchange.com/questions/239317/find-conditional-expectation-from-a-3-dimensional-random-vector?rq=1

function sample = gaussian_conditional_sample_ND(mu, Sigma, sampling_idx, observ)
    if sampling_idx == 1
        % Break covariance matrix down into quadrants
        sigma11 = Sigma(sampling_idx,sampling_idx);
        sigma12 = Sigma(sampling_idx, sampling_idx+1:end);
        sigma21 = Sigma(sampling_idx+1:end, sampling_idx);
        sigma22 = Sigma(sampling_idx+1:end,sampling_idx+1:end);
        
        mu_bar = mu(sampling_idx) + sigma12*inv(sigma22)*(observ-mu(sampling_idx+1:end))';
        Sigma_bar = sigma11 - sigma12*inv(sigma22)*sigma21;
        % Generating sample from this distribution
        sample = normrnd(mu_bar,Sigma_bar);
        
    else % change order of random variables
        % Change order of mu
        new_mu = mu;
        new_mu(sampling_idx) = []; % delete entry at sampling index
        new_mu = cat(2, mu(sampling_idx), new_mu)'; 
        mu = new_mu;
        % place it on top of the vector and transpose to get back a column vector

        % Changing order of Sigma
        new_Sigma = Sigma;
        new_Sigma(sampling_idx, :)= []; % removing row at sampling index
        new_Sigma(:, sampling_idx) = []; % removing column at sampling index
        concerned_row = Sigma(sampling_idx, :); % isolating this row
        concerned_row(sampling_idx) = [];
        new_Sigma = cat(1, concerned_row, new_Sigma); % placing row on top of matrix
        concerned_column = Sigma(:, sampling_idx); % isolating column
        concerned_column(sampling_idx) = [];
        concerned_column = cat(1,Sigma(sampling_idx,sampling_idx), concerned_column);
        new_Sigma = cat(2, concerned_column, new_Sigma); % adding column as 1st column
        Sigma = new_Sigma;
        
        % Break covariance matrix down into quadrants
        sigma11 = Sigma(sampling_idx,sampling_idx);
        sigma12 = Sigma(sampling_idx, sampling_idx+1:end);
        sigma21 = Sigma(sampling_idx+1:end, sampling_idx);
        sigma22 = Sigma(sampling_idx+1:end,sampling_idx+1:end);
        
        mu_bar = mu(sampling_idx) + sigma12*inv(sigma22)*(observ-mu(sampling_idx+1:end))';
        Sigma_bar = sigma11 - sigma12*inv(sigma22)*sigma21;
        % Generating sample from this distribution
        sample = normrnd(mu_bar,Sigma_bar);
    end
end
