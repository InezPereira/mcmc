%% Leapfrog algorithm
% Also inspired from implementations from Radford Neal, including from his
% GRIMS R package and/or formulation from Chapter 5 of the "Hanbook of
% Markov Chain Monte Carlo.

        % Leapfrog algorithm
function [q, p] = leapfrog_temp(p, epsilon, q, U, grad_U, L, Sigma, temp)
    % Check validatity of temp(erature) argument
    if temp <=0
        error('The temperature argument has to be a positive number.');
    end
    if length(p)~=length(q) || length(q)~=length(temp)
        error('The temperature argument has to have the same length as q.');
    end
    
    % Warn user of possibly inappropriate temp values
    if temp>2
        disp("The leapfrog algorithm will be run, but be warned that that temperature value seems a bit too high.")
    end
    
    % Depending on L, define the leapfrog steps.
    if mod(L,2)==0
        for ii=1:L
            % Before leapfrog step: multiply or divide by temperature, 
            % depending on which half of the trajectory is considered.
            if ii <= L/2
                p = p.*sqrt(temp);
            elseif ii>L/2
                p = p./sqrt(temp);
            end
            
        % Make half a step for momentum
        p = p - epsilon/2.*grad_U(q, U)'; % gradient is taken with respect to every q_i
        % Alternate full steps for position and momentum variables

        % Full step for position
        q = q + epsilon.*p./Sigma;
        
        % Make another half step for the momentum in the end
        p = p - epsilon.*grad_U(q, U)'/2;
        
        % After leapfrog step: multiply or divide by temperature, 
        % depending on which half of the trajectory is considered. 
        if ii <= L/2
            p = p.*sqrt(temp);
        elseif ii>L/2
            p = p./sqrt(temp);
        end
        
        end
        
        if mod(L,2)==1
        for ii=1:L
            % Before leapfrog step: multiply or divide by temperature, 
            % depending on which half of the trajectory is considered.
            if ii <= ceil(L/2)-1
                p = p.*sqrt(temp);
            elseif ii == ceil(L/2)
                p = p.*sqrt(temp);
            elseif ii>L/2
                p = p./sqrt(temp);
            end
            
        % Make half a step for momentum
        p = p - epsilon/2.*grad_U(q, U)'; % gradient is taken with respect to every q_i
        % Alternate full steps for position and momentum variables

        % Full step for position
        q = q + epsilon.*p./Sigma;
        
        % Make another half step for the momentum in the end
        p = p - epsilon.*grad_U(q, U)'/2;
        
        % After leapfrog step: multiply or divide by temperature, 
        % depending on which half of the trajectory is considered. 
        if ii <= ceil(L/2)-1
            p = p.*sqrt(temp);
        elseif ii == ceil(L/2)
            p = p./sqrt(temp);
        elseif ii>L/2
            p = p./sqrt(temp);
        end
        
        end


     
    end
end