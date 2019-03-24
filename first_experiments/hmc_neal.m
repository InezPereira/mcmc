%% Hamiltonian MC using Neal's Chapter from the Handbook of MCMC

% Heavily inspired on Radford N. Neal's chapter "MCMC using
% Hamiltonian Dynamics" (from the Handbook of Markov Chain Monte Carlo)

% Distribution of the momentum variables:
% - ﻿We can choose the distribution of the momentum variables, p, which 
% are independent of q, as we wish, specifying the distribution via the 
% kinetic energy function, K(p). Current practice with HMC is to use a 
% quadratic kinetic energy, as in Equation 5.5, which leads p to have a 
% zero-mean multivariate Gaussian distribution. Most often, the components
% are specified to be independent. So you retrieve the energy function in
% equation 5.23

%  Input arguments:
%     - U: function which returns the potential energy given a value for q
%     - grad_U : returns the vector of partial derivatives of U given q
%     - epsilon: stepsize for leapfrog steps
%     - L : number of leapfrog steps in the trajectory
%     - current_q : current position that the trajectory starts from
%     - mu, Sigma: moments of the distribution of p (which is often assumed gaussian)


function sample = hmc_neal(U, grad_U, mu, Sigma, epsilon, L, current_q)
    q = current_q;

    % In the first step, new values for the momentum are drawn from their
    % distribution (in practise often a gaussian distribution)
    p = normrnd(mu, Sigma);  % You want to generate one sample p_i per mu or Sigma.
    current_p = p;
    
    % Leapfrog algorithm
    % Make half a step for momentum
     p = p - epsilon/2*grad_U(q); % gradient is taken with respect to every q_i
     % Alternate full steps for position and momentum variables
     for ii=1:L
         % Full step for position
         q = q + epsilon*p./Sigma;
         
         % Make full step for the momentum, except at the end of trajectory
         if ii~=L
             p = p - epsilon*grad_U(q);
         end
     end
     
     % Make another half step for the momentum in the end
     p = p - epsilon*grad_U(q)/2;
     
     % Negate momentum to make proposal symmetric
     p = -p;
     
     % Evaluate potential and kinetic energies at start and end of the
     % trajectory
     current_U = U(current_q);
     current_K = sum(current_p.^2./Sigma)/2;
     proposed_U = U(q);
     proposed_K = sum(p.^2./Sigma)/2;
     
     % Compute acceptance probability
     alpha = exp(-proposed_U+current_U-proposed_K+current_K);
     
     if alpha > rand()
         sample = q; % accept
     else
         sample = current_q; % reject
     end
