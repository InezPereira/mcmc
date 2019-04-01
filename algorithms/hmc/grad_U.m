function value = grad_U(q, p_tilde)
    value = gradient_ND(q, @(q) U(q, p_tilde));
end
