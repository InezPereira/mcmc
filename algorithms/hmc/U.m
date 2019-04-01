function potentialEnergy = U(q, p_tilde)
    if ~isvector(q)
        error('Input q has to be a vector.')
    else
        potentialEnergy = sum(-log(p_tilde(q)));
    end
end