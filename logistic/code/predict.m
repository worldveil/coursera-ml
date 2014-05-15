function p = predict(theta, X)
    %PREDICT Predict whether the label is 0 or 1 using learned logistic 
    %regression parameters theta
    %   p = PREDICT(theta, X) computes the predictions for X using a 
    %   threshold at 0.5 (i.e., if sigmoid(theta'*x) >= 0.5, predict 1)
    
    % we can tweak this for an AUC curve
    THRESHOLD = 0.5;
    
    % probably some better syntatic sugar here...
    p = sigmoid(X * theta);
    p(p >= THRESHOLD) = 1;
    p(p < THRESHOLD) = 0;
    
end
