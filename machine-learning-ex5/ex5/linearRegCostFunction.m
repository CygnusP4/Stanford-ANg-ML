function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the
%   cost of using theta as the parameter for linear regression to fit the
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
% =========================================================================

err        = ((X * theta) - y);
err_sq     = (err .^ 2);
theta_sq   = (theta(2:end,:) .^ 2);
J          = (0.5/m) * (sum(err_sq(:)) + (lambda * sum(theta_sq(:))));

unreg_grad      = (1/m) * (X' * err);
reg_vector      = (lambda/m) * theta;
reg_vector(1,:) = 0;
grad            = unreg_grad + reg_vector;

grad = grad(:);

end
