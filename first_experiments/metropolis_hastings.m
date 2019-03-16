%% Importing all the necessary functions
addpath('../');

%% Create strange-ish distribution
x = [-30:0.1:40];
y1 = normpdf(x, 0, 2);
y2 = normpdf(x, 20, 7);
p = create_distribution(y1,y2)

% fun = @create_distribution;
% AUC = integral(@(y1,y2)fun(y1,y2), -Inf, Inf)

figure(1)
plot(x, p)
xlabel('x')
ylabel('p(x)')
title('Function we want to approximate')

%% Create unnormalized distribution and proposal distribution

p_tilde = p * 2;
sigma = 1;
k = 1;

n_iterations=2000;

% How to define the proposal distribution?
x0 = median(x); % Pure intuition. Not backed up by anything. 
% In this example, however, it would lead to less rejection, should we do
% something like rejection sampling.
[max_p, idx] = max(p);
x0 = x(idx); % based on the Laplace Approximation (Bishop p. 214)

% Sigma cannot be defined arbitrarily! At least not for the Laplace
% approximation.

q = normpdf(x, x0, sigma);

% Plot the functions before guaranteeing that q > p
figure(2) 
plot(x, p_tilde)
hold on
plot(x, q)

% Guarantees that k*q > p 
iter = 1
while sum(p_tilde > k*q) > 1 % does the job but will be inefficient!
    k = k + 1;
    sigma = sigma + 1
    q = normpdf(x, x0, sigma)
    iter = iter + 1;
    k
    sigma
end

iter = 1
% From graphical intuition
while sum(p_tilde > k*q) > 1
    if max(p_tilde)>max(k*q)
        k = k + 1;
        iter = iter + 1;
    else 
        sigma = sigma + 1;
        iter = iter + 1;
    end
    q = normpdf(x, x0, sigma)
    sigma
    k
    iter
end


q_prop = k*normpdf(x, x0, sigma)

figure(3) 
plot(x, p_tilde)
hold on
plot(x, q_prop)


% 
% for i = 1:n_iterations
%     mean = 
%     q = normpdf()
% end

%% Getting to the Metropolis-Hastings algorithm

% Creating the proposal distribution
% To figure out how to define the conditional distribution of two gaussian
% random variables: https://stats.stackexchange.com/questions/30588/deriving-the-conditional-distributions-of-a-multivariate-normal-distribution
% Also: https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Conditional_distributions
function q = prop_fun(x1, x2, mean1)

% Initialization, at t=0:
z_t = median(x)

% t=1:
z_star = 

accept_prob = min(1, p_tilde(z_star)/p_tilde(z_t))

if accept_prob > rand
    z_t = z_star


