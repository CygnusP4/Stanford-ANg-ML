function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
% =============================================================

prediction = sigmoid(X * theta);
% Remove the first theta which shouldn't be regularized.
reg_J      = (lambda/(2*m)) * (sum(theta .^ 2) - (theta(1:1) .^ 2));
J = (-1/m) * ((y' * log(prediction)) + ((1-y)' * log(1-prediction))) + reg_J;

num_features = size(X, 2);

err      = (prediction - y);
delta    = [];
for j = 1:num_features
    d        = sum((X'(j,:)) * err);
    delta    = cat(1,delta,d);
end

k = size(theta);
% zero out the first theta which shouldn't be regularized.
reg_theta = cat(1,[0],theta(2:k,:));
grad      = ((1/m) * delta) + ((lambda/m) * reg_theta);

end
