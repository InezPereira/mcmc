%% Gibbs Sampling
%% Standard import statements
addpath('../');

%% Preliminary experiments

mu = [0 0];
Sigma = [.25 .3; .3 1];
x1 = -3:.2:3; x2 = -3:.2:3;
[X1,X2] = meshgrid(x1,x2);
F = mvnpdf([X1(:) X2(:)],mu,Sigma);
F = reshape(F,length(x2),length(x1));

figure(1);
surf(x1,x2,F);
caxis([min(F(:))-.5*range(F(:)),max(F(:))]);
axis([-3 3 -3 3 0 .4])
xlabel('x1'); ylabel('x2'); zlabel('Probability Density');
colorbar;

%% Gibbs sampling - special case of Metropolis-Hastings
% Strong dependencies between successive samples.

% Check that the Sigma matrix is psd
all(eig(Sigma) > 0)

% Generate first sample
x0 = mvnrnd(mu, Sigma);

% Define number of iterations
n_iter = 1000000

x = {x0}
for ii=1:n_iter
    % Generate sample x1 from p(x1|x2=a)
    sample_x1 = gaussian_conditional_sample(mu, Sigma, 1, 2, x{ii}(2));
    % Generate sample x2 from p(x2|x1=a)
    sample_x2 = gaussian_conditional_sample(mu, Sigma, 2, 2, x{ii}(1));
    % Store samples
    x{ii+1} = [sample_x1, sample_x2];
end

% Extracting the several dimensions
x1_samples = ones(1, length(x)); % pre-allocation
for ii=1:length(x)
    x1_samples(1,ii) = x{ii}(1);
end 

x2_samples = ones(1, length(x)); % pre-allocation
for ii=1:length(x)
    x2_samples(1,ii) = x{ii}(2);
end 

% Create concatenated matrux for plotting
concat_mat = cat(1,x1_samples, x2_samples).';

% Plotting
figure(2)
hist3(concat_mat, 'Nbins', [40, 40], 'FaceColor', [100 149 237]/255) %, 'FaceAlpha', 0.5)
title('Histogram of generated samples')
xlabel('x1'); ylabel('x2'); zlabel('Absolute frequency');

%% Gibbs sampling for > 3 dimensions

% Create new multivariate Gaussian
mu = [0 0 0];

% CHALLENGE: how to generate large psd matrices?
% https://math.stackexchange.com/questions/2624199/how-to-generate-a-large-psd-matrix-a-in-mathbbrn-times-n-where-math
v = [.25 .3 2 7 .6 .2;.67 .43 3 8 .6 .1;1 .7 1 2 .7 .2]
Sigma = v*v'
% Sigma = [.25 .3 2; .3 1 .1; 2 .1 5];
x1 = -3:.2:3; x2 = -3:.2:3; x3 = -3:.2:3

% Check that the Sigma matrix is psd
all(eig(Sigma) > 0)

% Generate first sample
x0 = mvnrnd(mu, Sigma);

% Define number of iterations
n_iter = 1000

% Generate samples
x = {x0};
for ii=1:n_iter
    % Generate sample x1 from p(x1|x2=a)
    observ = [x{ii}(2),x{ii}(3)]
    sample_x1 = gaussian_conditional_sample_ND(mu, Sigma, 1, observ);
    % Generate sample x2 from p(x2|x1=a)
    observ = [x{ii}(1),x{ii}(3)]
    sample_x2 = gaussian_conditional_sample_ND(mu, Sigma, 2, observ);
    % Same for x3
    observ = [x{ii}(1),x{ii}(2)]
    sample_x3 = gaussian_conditional_sample_ND(mu, Sigma, 3, observ);
    % Store samples
    x{ii+1} = [sample_x1, sample_x2];
end



