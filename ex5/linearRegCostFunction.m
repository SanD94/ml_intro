function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples



% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%


% cost function
a1 = X;
z2 = a1 * theta;
a2 = z2;
J = 1/(2*m) * sum((a2 - y).^2) + lambda/(2*m) * sum(theta(2:end).^2);

% gradient
grad = mean((a2 - y) .* X) + lambda/m * theta';
grad(1) = grad(1) - lambda/m * theta(1);



% =========================================================================



grad = grad(:);
end
