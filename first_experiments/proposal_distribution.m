function y = proposal_distribution(x, mu, sigma, k)
    y = k*normpdf(x, mu, sigma)
end