%% Demo script to run Hamiltonian MC in 2D

%% Import statements
addpath('../');
addpath('util/');
addpath('hmc/');
addpath('diagnostics/');
% addpath('/Users/ines/Documents/MATLAB/Euler');

%% Define the parameters necessary for kinetic energy function (assumed gaussian)
n_dim = 2;
mu = repmat(0, 1, n_dim);
Sigma = repmat(1, 1, n_dim); % Sigma for each variable p_i, here considered to have 0 covariance

%% For the potential energy
% Assume you know the posterior up to the normalization constant and that
% it is a gaussian mixture
% mu_tilde = {[-5, -5], [0,0], [3,3]};
n_mix = 3
mu_tilde = {repmat(-10, 1, n_dim), repmat(0, 1, n_dim), repmat(3, 1, n_dim)}
sigma_tilde = {2*eye(n_dim), 3*eye(n_dim), 4*eye(n_dim)}; % Assumption of independence made (no covariance)
weights = repmat(1/n_mix, 1, n_mix);
p_tilde = @(X) gaussian_mix_ND(X, mu_tilde, sigma_tilde, weights);
U = @(q) sum(-log(p_tilde(q)));
spacing = 1
step = 2
grad_U = @(q, U)subsref(gradient(U(q)-step:spacing:U(q)+step), struct('type', '()', 'subs', {{ceil(length(U(q)-step:spacing:U(q)+step)/2)}}));


%% Initial plots of the distribution we want to approximate
x = -15:.1:10; %// x axis
y = -15:.1:10; %// y axis

[X, Y] = meshgrid(x,y);
input = [X(:) Y(:)];
Z = p_tilde(input);
Z = reshape(Z,size(X)); %// put into same size as X, Y

figure(1)
surf(X,Y,Z) %// 3D plot

figure(2)
contour(X,Y,Z), axis equal  %// contour plot; set same scale for x and y...

%% Other parameters for the HMC (leapfrog) algorithm
epsilon = 0.1; % Neal,  p. 141: " We must make epsilon proportional to d^{-1/4} to maintain a reasonable acceptance rate.
L = 10;
q0 = {[-20,-10], [0,0], [40,0]}
reject = [0, 0, 0];
nChains = 3;
% temp = repmat(1.1, 1, nChains);
temp = [10,100,200];
s   amples = {q0(1), q0(2), q0(3)};
n_iter = 10^4;

tic
% ticBytes(gcp);
for jj=1:n_iter
    % Run the Hamiltonian algorithm
    for ii = 1:nChains
        [samples{ii}{jj+1}, reject(ii)] = hmc_neal(U, grad_U, mu, Sigma, epsilon, 1, L, 1, samples{ii}{jj}, reject(ii));

        if mod(jj,1000) ==0
            reject_rate = reject/jj;
            fprintf('Performing iteration number: (%d)\n', jj)
            fprintf('Current rejection rate: (%d)\n', reject_rate)
        end
        
%         if jj == n_iter
%             final_reject(ii) = reject / n_iter;
%         end
    end
%     
%     % State swapping step
%     for ii = 1:nChains
%         if ii == 1
%             kk = 2;
%         elseif ii == nChains
%             kk = nChains - 1;
%         else
%             beta = rand();
%             if beta < 0.5
%                 kk = ii - 1;
%             elseif beta > 0.5
%                 kk = ii + 1;
%             end
%         end
%         
%         % Compute probability of swapping
%         current_U = U(samples{ii}{jj+1});
% %         current_K = sum(current_p.^2./Sigma)/2;
%         proposed_U = U(samples{kk}{jj+1});
% %         proposed_K = sum(p.^2./Sigma)/2;
%         gamma = min(1, exp(current_U - proposed_U)*(1/temp(ii)-1/temp(kk)));
%         
%         if gamma > rand()
%             samples{ii}{jj+1} = samples{kk}{jj+1};
%         end
%     end
% From "Parallel tempering: Theory, applications and new perspective"
% ﻿The acceptance probability in Hamiltonian parallel tempering for a swap 
% between replicas i and j is given by
end
% tocBytes(gcp)
toc

% samples_first_half= samples;
% samples = samples(randperm(nChains));
% 
% for ii = 1:nChains
%         for jj=ceil(n_iter)/2:n_iter
%         [samples{ii}{jj+1}, reject] = hmc_temp_traj(U, grad_U, mu, Sigma, epsilon, 1, L, 1, samples{ii}{jj}, reject, temp);
% 
%         if mod(jj,1000) ==0
%             reject_rate = reject/jj;
%             fprintf('Performing iteration number: (%d)\n', jj)
%             fprintf('Current rejection rate: (%d)\n', reject_rate)
%         end
%         
%         if jj == n_iter
%             final_reject(ii) = reject / n_iter;
%         end
%         end
% end


save('samples','samples')

%% Final visualizations

% Choose a chain:
chain = 1

% Figure to get the exploration space
figure(3)
contour(X,Y,Z), hold on
samples_plotted = cell2mat(samples{chain}')
samples_plotted = samples_plotted(1:10^2:end,:)
c = linspace(1,10,length(samples_plotted)); % more yellow is further along the line
% c2 = linspace(1,10,length(samples_plotted));
% c3 = linspace(1,10,length(samples_plotted));
% c = [c1; c2; c3]'
scatter(samples_plotted(:,1), samples_plotted(:,2), [], c);
line(samples_plotted(:,1), samples_plotted(:,2)) %, 'Color', c);
% plot(samples_plotted(:,1), samples_plotted(:,2), '-o', 'MarkerEdge', 'r');
title('Exploration of 2D Gaussian mixture by HMC')
hold off;


figure(4)
hist3(cell2mat(samples{chain}'), 'FaceColor', [100 149 237]/255, 'Nbins',[20,20]);
title("Hamiltonian Monte Carlo");
xlabel('X')
ylabel('Y')
zlabel('Absolute frequency');
savefig('Hybrid_MC')

figure(5)
traceplots(nChains, samples, 1)