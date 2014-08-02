function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
    %% GRADIENTDESCENTMULTI Performs gradient descent to learn theta
    %   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
    %   taking num_iters gradient steps with learning rate alpha
    [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters);
end
