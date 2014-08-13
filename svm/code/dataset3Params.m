function [C, sigma] = dataset3Params(X, y, Xval, yval)
    %% EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
    %where you select the optimal (C, sigma) learning parameters to use for SVM
    %with RBF kernel
    %   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
    %   sigma. You should complete this function to return the optimal C and 
    %   sigma based on a cross-validation set.
    %

    % ====================== YOUR CODE HERE ======================
    % Instructions: Fill in this function to return the optimal C and sigma
    %               learning parameters found using the cross validation set.
    %               You can use svmPredict to predict the labels on the cross
    %               validation set. For example, 
    %                   predictions = svmPredict(model, Xval);
    %               will return the predictions on the cross validation set.
    %
    %  Note: You can compute the prediction error using 
    %        mean(double(predictions ~= yval))
    %
    c_values = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
    sigma_values = c_values(:);
    best_accuracy = 0.0;
    
    for i = 1 : length(c_values)
        
        for j = 1 : length(sigma_values)
            
            % test this grid entry on our validation set 
            test_C = c_values(i);
            test_sigma = sigma_values(j);
            model = svmTrain(X, y, test_C, @(x1, x2) gaussianKernel(x1, x2, test_sigma));
            predictions = svmPredict(model, Xval);
            accuracy = sum(predictions == yval) / length(yval);
            
            % is this the best performing so far?
            if accuracy > best_accuracy
                best_accuracy = accuracy;
                C = test_C;
                sigma = test_sigma;
            end
        end 
    end
    
    fprintf('Best hyperparameters were C = %f, sigma = %f => best cv acc = %f\n', ...
        C, sigma, best_accuracy);
    
end
