%% Function to compute the state swapping step of parallel tempering for HMC.
function samples = state_swapping(iter, nChains, samples, temp, current_U, current_K)
    % State swapping step
    % This first part comes from the book "Advanced Markov Chain Monte 
    % Carlo Methods: Learning from Past Samples" Section 5.4 on parallel 
    % tempering
    for ii = 1:nChains
        if ii == 1
            kk = 2;
        elseif ii == nChains
            kk = nChains - 1;
        else
            beta = rand();
            if beta < 0.5
                kk = ii - 1;
            elseif beta > 0.5
                kk = ii + 1;
            end
        end
        
        % Compute probability of swapping
        % Formula for Hamiltonian parallel tempering taken from Earl et al. 
        % "Parallel tempering: Theory, applicationas, and new perspectives."
        % In this article and in the article by ﻿Fukunishi et al., it is
        % assumed that beta, the inverse temperature (approx.) stays the
        % same for all replicas.
        
        %  If I don't assume beta to the same for all and derive the
        %  equations myself, I get:
        beta = temp.^(-1);
        initial_config = beta(ii)*(current_U(ii)+current_K(ii)) + beta(kk)*(current_U(kk)+current_K(kk));
        
        % Define proposed Hamiltonians
        proposed_config = beta(ii)*(current_U(kk)+current_K(kk)) + beta(kk)*(current_U(ii)+current_K(ii));
        
        alpha = min(1, exp(proposed_config - initial_config));
        
        if alpha > rand()
            samples{ii}{iter+1} = samples{kk}{iter+1};
        end
    end
     