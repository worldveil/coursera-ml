function [J, grad] = lrCostFunction(theta, X, y, lambda)
    %% LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
    %regularization
    %   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
    %   theta as the parameter for regularized logistic regression and the
    %   gradient of the cost w.r.t. to the parameters. 
    
    n = length(y); % number of training examples
    
    % cost: J, this time with the penalty for the magnitude of theta
    J = -1/n * sum(...
        y      .* log(sigmoid(X * theta)) + ...
        (1 - y) .* log(1 - sigmoid(X * theta)) ...
    ) + lambda / (2 * n) * sum(theta(2:end) .* theta(2:end));

    % gradient: compute as the derivative of the cost function
    grad = 1/n * X' * ((sigmoid(X * theta)) - y) + theta * lambda / n;
   
    % we do not regularize the constant offset term, 
    % undo the gradient penalization for the constant x_0 term
    grad(1) = grad(1) - lambda / n * theta(1);
end
