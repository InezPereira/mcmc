% Playlist on OOP in Matlab: https://www.youtube.com/watch?v=DBTSMDv7pKI&list=PL4bFPZnVfMSVHohsYUmAdLUvTpxCTVPyN&index=1

classdef hmcClass
    properties
        % Basic parameters and results
        n_dim = 2
        p_tilde % Unnormalized target distribution
        nChains = 2
        n_iter = 10^3
        samples = {}
        q0 = {{[0,0]}, {[1,1]}}
       
        % Parameters for leapfrog algorithm
        epsilon = 0.1
        randomize_epsilon = 1
        L = 20
        randomize_L = 1

    end
    
    properties (Dependent)
        temp 
        reject
        
        % Potential energy function
        U
        grad_U
        
        % Kinetic energy function
        Sigma_K     % Zero covariance assumed
        mu_K        % Zero mean by default
        K % Assume kinetic energy is gaussian
        
    end
    
    methods
        % Set functions for class properties
        function obj = set.n_dim(obj, n_dim)
            if n_dim > 0
                obj.n_dim = floor(n_dim);
            else
                error('The number of dimensions must be a positive integer.')
            end
        end
        
        function obj = set.nChains(obj, nChains)
            if nChains > 0
                obj.nChains = floor(nChains);
            else
                error('nChains must be a positive integer.')
            end
        end
        
        function obj = set.p_tilde(obj, p_tilde)
            if isa(p_tilde, 'function_handle')
                obj.p_tilde = p_tilde;
            else
                error('p_tilde must be a distribution and hence a function.')
            end  
        end
        
        function obj = set.q0(obj, q0)
            if isvector(q0) && length(q0{1})==obj.n_dim && length(q0)==obj.nChains
                obj.q0 = q0;
            elseif length(q0{1})~=obj.n_dim
                error('q0 must contains vectors with dimensions equal to n_dim.')
            elseif length(q0)~=obj.nChains
                error('q0 must be a cell array with length equal to nChains.')
            end  
        end
        
        function obj = set.n_iter(obj, n_iter)
            if n_iter > 0
                obj.n_iter = floor(n_iter);
            else
                error('The number of iterations must be a positive integer.')
            end
        end
        
        function obj = set.U(obj, U)
            if isa(U, 'function_handle')
                obj.U = U;
            else
                error('U, the potential energy, must be a function.')
            end
        end
        
        function obj = set.epsilon(obj, epsilon)
            if isnumeric(epsilon) && epsilon > 0
                obj.epsilon = epsilon;
            else
                error('Epsilon must be a positive numeric.')
            end
        end
        
        function obj = set.L(obj, L)
            if L > 0
                obj.L = floor(L);
            else
                error('L must be a positive integer.')
            end
        end
        
        % Get dependent properties
        function temp = get.temp(obj)
            temp = ones(1, obj.nChains);
        end
        
        function reject = get.reject(obj)
            reject = zeros(1,obj.nChains);
        end

        function U = get.U(obj)
            U = @(q) sum(-log(obj.p_tilde(q)));
        end

        function grad_U = get.grad_U(obj)
            grad_U = @(q) gradient_ND(q, obj.U);
        end

        function Sigma_K = get.Sigma_K(obj)
            Sigma_K = ones(1, obj.n_dim);
        end
        
        function mu_K = get.mu_K(obj)
            mu_K = zeros(1, obj.n_dim);
        end
        
        function K = get.K(obj)
            K = @(p) sum((p-obj.mu_K).^2./obj.Sigma_K)/2; % Assume kinetic energy is gaussian        
        end
                
        
        % Actual methods
        function obj = hmc(obj)
            for ii=1:obj.nChains
                obj.samples{ii} = obj.q0{ii};
            end
            for jj=1:obj.n_iter
                for ii = 1:obj.nChains
            % Run the Hamiltonian algorithm
                    [obj.samples{ii}{jj+1}, obj.reject(ii)] = hmc(obj.U, obj.grad_U, obj.mu_K, obj.Sigma_K, obj.epsilon, obj.randomize_epsilon, obj.L, obj.randomize_L, obj.samples{ii}{jj}, obj.reject);
                    if mod(jj,10) ==0
                        reject_rate = obj.reject/jj;
                        fprintf('In chain number: (%d)\n', ii)
                        fprintf('Performing iteration number: (%d)\n', jj)
                        fprintf('Current rejection rate: (%d)\n', reject_rate)
                    end
                end
            end
        end

        
    end
    
   
end