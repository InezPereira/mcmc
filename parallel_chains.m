%% Demo script to run Hamiltonian MC in 2D

%% Import statements
addpath('../');
addpath('util/');
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
final_reject = []

tic
ticBytes(gcp);
nChains = 3;
samples = {};
reject = 0
parfor ii = 1:nChains
% Run the Hamiltonian algorithm
    n_iter = 10^4;
    samples{ii}= {cell2mat(q0(ii))};
    reject = 0
    for jj=1:n_iter
        [samples{ii}{jj+1}, reject] = hmc_neal(U, grad_U, mu, Sigma, epsilon, 1, L, 1, samples{ii}{jj}, reject);

        if mod(jj,1000) ==0
            reject_rate = reject/jj;
            fprintf('Performing iteration number: (%d)\n', jj)
            fprintf('Current rejection rate: (%d)\n', reject_rate)
        end
        
        if jj == n_iter
            final_reject(ii) = reject / n_iter
        end
    end
end
tocBytes(gcp)
toc

% samples_hmc_mat = cell2mat(samples)
% save('samples_hmc_mat','samples_hmc_mat')

%% Final visualizations

% Choose a chain:
chain = 3

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


figure(4)
hist3(cell2mat(samples{chain}'), 'FaceColor', [100 149 237]/255, 'Nbins',[20,20]);
title("Hamiltonian Monte Carlo");
xlabel('X')
ylabel('Y')
zlabel('Absolute frequency');
savefig('Hybrid_MC')