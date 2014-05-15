function [J, grad] = costFunction(theta, X, y)
    %COSTFUNCTION Compute cost and gradient for logistic regression
    %   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
    %   parameter for logistic regression and the gradient of the cost
    %   w.r.t. to the parameters.

    % Initialize some useful values
    n = length(y); % number of training examples
    
    % cost: J
    J = -1/n * sum(...
        y      .* log(sigmoid(X * theta)) + ...
        (1 - y) .* log(1 - sigmoid(X * theta)) ...
    );

    % gradient: compute as the derivative of the cost function
    grad = 1/n * X' * ((sigmoid(X * theta)) - y);

end
