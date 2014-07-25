function [J, grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
    %NNCOSTFUNCTION Implements the neural network cost function for a two layer
    %neural network which performs classification
    %   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
    %   X, y, lambda) computes the cost and gradient of the neural network. The
    %   parameters for the neural network are "unrolled" into the vector
    %   nn_params and need to be converted back into the weight matrices. 
    % 
    %   The returned parameter grad should be a "unrolled" vector of the
    %   partial derivatives of the neural network.
    %

    % Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
    % for our 2 layer neural network
    Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                     hidden_layer_size, (input_layer_size + 1));

    Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                     num_labels, (hidden_layer_size + 1));
    
    n = size(X, 1); 
    Theta1_grad = zeros(size(Theta1));
    Theta2_grad = zeros(size(Theta2));

    % ====================== YOUR CODE HERE ======================
    % Instructions: You should complete the code by working through the
    %               following parts.
    %
    % Part 1: Feedforward the neural network and return the cost in the
    %         variable J. After implementing Part 1, you can verify that your
    %         cost function computation is correct by verifying the cost
    %         computed in ex4.m
    %
    % Part 2: Implement the backpropagation algorithm to compute the gradients
    %         Theta1_grad and Theta2_grad. You should return the partial derivatives of
    %         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
    %         Theta2_grad, respectively. After implementing Part 2, you can check
    %         that your implementation is correct by running checkNNGradients
    %
    %         Note: The vector y passed into the function is a vector of labels
    %               containing values from 1..K. You need to map this vector into a 
    %               binary vector of 1's and 0's to be used with the neural network
    %               cost function.
    %
    %         Hint: We recommend implementing backpropagation using a for-loop
    %               over the training examples if you are implementing it for the 
    %               first time.
    %
    % Part 3: Implement regularization with the cost function and gradients.
    %
    %         Hint: You can implement this around the code for
    %               backpropagation. That is, you can compute the gradients for
    %               the regularization separately and then add them to Theta1_grad
    %               and Theta2_grad from Part 2.
    % =========================================================================

    
    %================PART 1================%
    % Feedforward to find the cost of the network given
    % the examples and theta matrices.
    
    % add bias units
    X = [ones(n, 1) X];
    
    % mutliply to get output activations and linear combination z's
    z2 = X * Theta1';
    a2 = sigmoid(z2);
    a2 = [ones(n, 1) a2];
    z3 = a2 * Theta2';
    a3 = sigmoid(z3);
    hthetax = a3;
    
    % expand the y vector into a binary vector with 
    % only a single 1 set
    y_expanded = zeros(n, num_labels);
    for i = 1 : n
        y_expanded(i, y(i)) = 1;
    end
    
    % for each h(x), compute contribution to cost
    % TODO: vectorize this!
    cost = 0;
    for i = 1 : n  % for each example
        for k = 1 : num_labels  % for each class           
            cost = cost - ...
                y_expanded(i, k) * log(hthetax(i, k)) - ...
                (1 - y_expanded(i, k)) * log(1 - hthetax(i, k));
        end
    end
    J = cost / n;
    
    %================PART 2================%
    % Implement backpropagation and fill the grad
    % vector with the unrolled matrices
    
    Delta2 = zeros(size(Theta2, 1), size(Theta2, 2));
    Delta1 = zeros(size(Theta1, 1), size(Theta1, 2));
    
    for t = 1 : n
        a1 = X(t, :);
        delta3 = a3(t, :) - y_expanded(t, :);
        delta2 = (Theta2' * delta3') .* a2(t, :)' .* (1 - a2(t, :)');
        Delta2 = Delta2 + delta3' * a2(t, :);
        Delta1 = Delta1 + delta2(2:end) * a1;
    end
    
    % don't use the theta bias terms, set them to 0
    copyTheta1 = Theta1;
    copyTheta1(:, 1) = 0;
    copyTheta2 = Theta2;
    copyTheta2(:, 1) = 0;
    
    % set accumulators equal to sum plus the regularization matrix, divided
    % by the number of examples
    Delta2 = (Delta2 + lambda * copyTheta2) / n;
    Delta1 = (Delta1 + lambda * copyTheta1) / n;
    
    % Gradients:
    % note the size of each gradient includes an extra +1 for the
    % bias term
    grad = [Delta1(:) ; Delta2(:)];
    
    %================PART 3================%
    % Add in the regularization term to the cost
    
    % compute regularization term
    regularization = lambda / 2 / n * ( ... 
        sum(sum(Theta1(:, 2:end) .^ 2)) + ...
        sum(sum(Theta2(:, 2:end) .^ 2)) ...
    );
    
    % add in regularization
    J = J + regularization;
    
end
