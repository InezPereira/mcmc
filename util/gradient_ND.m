function x_grad  = gradient_ND(x, func)

% Only works for two dimensions

if ~isvector(x)
    error('The point x has to be a vector!')
else
    
    % Create points around the point along the several dimensions
    P = cell(length(x), 1);
    for ii=1:length(x)
        P{ii} = x(ii)-2:1:x(ii)+2;
    end
    
    % Create meshgrid
    M = cell(length(x),1);
    [M{:}] = meshgrid(P{:});
    
    % Create compatible input
    % https://stackoverflow.com/questions/758736/how-do-i-iterate-through-each-element-in-an-n-dimensional-matrix-in-matlab
    input = zeros(numel(M{1}), length(x));
    for idx=1:numel(M{1})
        % https://ch.mathworks.com/matlabcentral/answers/90948-how-can-i-access-element-with-same-index-from-multiple-cells
        input(idx, :) = cellfun(@(c)c(idx), M);
    end

    % Pass everything through your function
    Z = func(input);
    
    % Reshape output
    Z = reshape(Z,size(M{1}));
    
    % Compute gradient
    nd = sum(size(Z)>1);
    G = cell(nd,1);
    [G{:}] = gradient(Z);
    
    % Find your point!!
    len = numel(G{1});
    x_grad = zeros(length(x),1);
    x_grad(:, 1) = cellfun(@(c)c(ceil(len/2)), G);
    
end
end
