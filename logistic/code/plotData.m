function plotData(X, y)
    %PLOTDATA Plots the data points X and y into a new figure 
    %   PLOTDATA(x,y) plots the data points with + for the positive examples
    %   and o for the negative examples. X is assumed to be a Mx2 matrix.

    % Create New Figure
    figure; hold on;

    % ====================== YOUR CODE HERE ======================
    % Instructions: Plot the positive and negative examples on a
    %               2D plot, using the option 'k+' for the positive
    %               examples and 'ko' for the negative examples.
    %

    % select positives and negatives
    negatives = find(y == 0);
    positives = find(y == 1);

    % plot as scatters
    scatter(X(positives, 1), X(positives, 2), 36, 'r', 'k+'); 
    scatter(X(negatives, 1), X(negatives, 2), 36, 'b', 'ko'); 
    title('University Admittance Based on Exam Scores');
    
    hold off;
end

