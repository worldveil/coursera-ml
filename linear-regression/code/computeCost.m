function J = computeCost(X, y, theta)
    %% COMPUTECOST Compute cost for linear regression
    %   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
    %   parameter for linear regression to fit the data points in X and y
    J = sum((X * theta - y) .^ 2) / 2 / length(y);
end