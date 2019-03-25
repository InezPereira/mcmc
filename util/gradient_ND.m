function [FX, FY] = gradient_ND(x, func)

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
    Z = func(input)
    
    
    % Reshape output
    
    % Compute gradient
    G = cell(nd,1);
    [G{:}] = gradient(Z);
    
    % ------------------------
    % Compute dimensions point and hence number of dimensions needed
    n_dim = length(point);

    % Create vector of numbers around that point. Distance of 1
    % These vector have to be created separately for each dimension and need to
    % match dimensions
    x = x(1)-10:1:x(1)+10;
    y = x(2)-10:1:x(2)+10;

    % Create meshgrid
    [X, Y] = meshgrid(x,y);
    input = [X(:) Y(:)];

    % Compute the output from the function whose gradient we wish to estimate
    Z = func(input);
    
    % Reshape output
    Z = reshape(Z,size(X));
    
    % Compute gradient

    
    point_grad = grad(find(x==point(1)), find(y==point(2)));
    
end
end
