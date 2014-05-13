function g = sigmoid(z)
    %SIGMOID Compute sigmoid functoon
    %   J = SIGMOID(z) computes the sigmoid of z.
    g = (exp(-z) + 1) .^ -1;
end
