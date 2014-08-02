function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
    %LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
    %regression with multiple variables
    %   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
    %   cost of using theta as the parameter for linear regression to fit the 
    %   data points in X and y. Returns the cost in J and the gradient in grad

    % Initialize some useful values
    n = length(y); % number of training examples
    grad = zeros(size(theta));

    % ====================== YOUR CODE HERE ======================
    % Instructions: Compute the cost and gradient of regularized linear 
    %               regression for a particular choice of theta.
    %
    %               You should set J to the cost and grad to the gradient.
    %
    
    % calculate the cost using the squared error and the complexity
    % regularization penalty
    scaled_squared_error = sum(((X * theta - y) .^ 2)) / 2 / n;
    regularization = lambda * sum(theta(2:end) .^ 2) / 2 / n;
    J = scaled_squared_error + regularization;
    
    % now caluclate the gradient for each entry of theta 
    error = X * theta - y;
    for j = 1 : length(theta)
        grad(j) = sum(error .* X(:, j)) / n;
        if j > 1
            grad(j) = grad(j) + lambda * theta(j) / n;
        end
    end
    
    grad = grad(:);  % unroll 
end
