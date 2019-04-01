%% Chapter 23 of Murphy's 
% "Machine Learning - a Probabilistic Perspective", 2012

% This script serves to implement the sampling methods discussed in the
% aforementioned chapter, complemented with other additional resources.

%% Import statements
addpath('~/Documents/MATLAB');
addpath('../');

%% Sampling from standard distributions - Using the cdf 
%(skipping the Box Muller method)

x = [-5:0.01:5];
p = normcdf(x, 0, 1); % returns the cdf of the standard normal distribution

figure(1)
plot(x, p)
xlabel('x')
ylabel('F(x)')
title('CDF of standard normal distribution')

% Generate observations of random variable following U(0,1)

U = rand([1,1000]);
s = norminv(U);

size_s = size(s);

figure(2)
histogram(s, 'FaceColor', [254,178,76]/255, 'NumBins', 100)
hold on

% And now how do I sample from my approximated standard gaussian given this
% histogram? 
% Check out: https://stats.stackexchange.com/questions/191725/sample-from-distribution-given-by-histogram

%% Rejection sampling - or simple alternative when the inverse cdf method cannot be used

% Create strange-ish distribution - Gaussian mixture of 2 gaussians
% x = [-30:0.1:40];
% y1 = normpdf(x, 0, 2);
% y2 = normpdf(x, 20, 7);
% p = create_distribution(y1,y2)

x = [-30:0.1:40]
mu_list = [0, 20];
sigma_list = [1, 7];
p = gaussian_mix(x, mu_list, sigma_list, weights);

% fun = @create_distribution;
% AUC = integral(@(y1,y2)fun(y1,y2), -Inf, Inf)

figure()
plot(x, p)
xlabel('x')
ylabel('p(x)')
title('Function we want to approximate')

% Create unnormalized distribution and proposal distribution
p_tilde = gaussian_mix(x, mu_list, sigma_list, weights);
plot(x, p_tilde)

% How to define the proposal distribution?
x0 = median(x); % Pure intuition. Not backed up by anything. 
% In this example, however, it would lead to less rejection, should we do
% something like rejection sampling.
[max_p, idx] = max(p_tilde);
x0 = x(idx); % based on the Laplace Approximation (Bishop p. 214)

% Sigma cannot be defined arbitrarily! At least not for the Laplace
% approximation.
sigma = 1;
k = 1;
q = normpdf(x, x0, sigma); % here, we are taking the biggest mode as the mean

% Plot the functions before guaranteeing that q > p_tilde
figure() 
plot(x, p_tilde)
hold on
plot(x, q)
title('Before adjusting the proposal distribution')

% Guarantees that k*q > p 
iter = 1;
sigma_final1 = sigma;
k1 = k;
while sum(p_tilde > k1*q) > 1 % does the job but will be inefficient!
    k1 = k1 + 1;
    sigma_final1 = sigma_final1 + 1
    q = normpdf(x, x0, sigma_final1)
    iter = iter + 1;
end
iter
k1
sigma_final1


iter = 1
sigma_final2 = sigma
k2 = k
% From graphical intuition
while sum(p_tilde > k2*q) > 1
    if max(p_tilde)>max(k2*q)
        k2 = k2 + 1;
        iter = iter + 1;
    else 
        sigma_final2 = sigma_final2 + 1;
        iter = iter + 1;
    end
    q = normpdf(x, x0, sigma_final2)
end
iter
k2
sigma_final2


q_prop1 = k1*normpdf(x, x0, sigma_final1)

figure() 
plot(x, p_tilde)
hold on
plot(x, q_prop1)
fill([x, fliplr(x)], [p_tilde, fliplr(q_prop1)], [250 128 114]/255)
alpha(0.5)


q_prop2 = k2*normpdf(x, x0, sigma_final2)


q_prop2 = proposal_distribution(x,x0, sigma_final2, k2)

figure() 
plot(x, p_tilde)
hold on
plot(x, q_prop2)
fill([x, fliplr(x)], [p_tilde, fliplr(q_prop2)], [176, 196, 222]/255)
alpha(0.5)

% Actually performing rejection sampling
% n_iterations=2000;
% for i = 1:n_iterations
%     r = -50 + 100.*rand(1,1000);
    n_points = 1000;
    r = normrnd(x0,sigma_final2, [1, n_points]); % First, generate number from q(z) (not kq(z))
    h = proposal_distribution(r, x0, sigma_final2, k2).*rand(1,n_points);
% end

% Your implementation follows Bishop more (see p. 529)
figure;
z_final = []
for i=1:length(h)
    % Select color
    if h(i) < gaussian_mix(r(i),mu_list,sigma_list, weights)
        mycolor = [34, 139, 34]/255;
        z_final = [z_final, r(i)];
    else
        mycolor = [178, 34, 34]/255;
    end
    % Plot the point
    scatter(r(i), h(i), 'MarkerEdgeColor', mycolor, 'MarkerFaceColor', mycolor)
    hold on
end
title('Rejection sampling - Accepted and rejected points')

figure;
histogram(z_final, 'Numbins', 20)
title('Histogram of the accepted z values') % we can now get an approximation to p from this!

%% Adaptive rejection sampling
% This method is supposed to come up with a tight upper envelope q(x) to
% any log concave density p(x).
% Examples of log concave functions: https://en.wikipedia.org/wiki/Logarithmically_concave_function
% A Gaussian mixture is not one of them. :(

% In Chapter 24, we will describe MCMC sampling, which is a more efficient 
% way to sample from high dimensional distributions. Sometimes this uses 
% (adaptive) rejection sampling as a subroutine, which is known as adaptive 
% rejection Metropolis sampling (Gilks et al. 1995).    

%% Importance sampling
% Unlike rejection sampling, we use all the samples.
% It's not really a sampling method. It's an approximation method.

