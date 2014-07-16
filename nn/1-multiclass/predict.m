function p = predict(Theta1, Theta2, X)
    %% PREDICT Predict the label of an input given a trained neural network
    %   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
    %   trained weights of a neural network (Theta1, Theta2)
    
    % add offset to each X vector
    n = size(X, 1);  % number of examples
    X = [ones(n, 1) X];
    
    % compute the activation of the first layer
    activation_layer_1 = sigmoid(Theta1 * X');
    
    % then add the offset as row of ones on top of activation matrix
    activation_layer_1 = [ones(1, size(activation_layer_1, 2)); activation_layer_1];
    
    % compute activation for layer two (our final output)
    activation_layer_2 = sigmoid(Theta2 * activation_layer_1);
    
    % transpose so each row is an example and its 10 activations, one for
    % each digit class
    output_activation = activation_layer_2';  % for clarity, NOT efficiency...
    
    % take the prediction for each to be the maximum activation for that
    % particular class
    [~, p] = max(output_activation, [], 2);
end
