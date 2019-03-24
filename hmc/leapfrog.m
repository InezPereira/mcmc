%% Leapfrog algorithm
        % Leapfrog algorithm
function [q, p] = leapfrog(p, epsilon, q, grad_U, L, Sigma)
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
end