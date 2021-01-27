function [J grad] = nnCostFunction(nn_params, ...
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

% Setup some useful variables
m = size(X, 1);

% You need to return the following variables correctly
J = 0;
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
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives
%         of the cost function with respect to Theta1 and Theta2 in Theta1_grad
%         and Theta2_grad, respectively. After implementing Part 2, you can check
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
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
% ============================================================================

% -------------------------------------------
% Calculate predictions for all layers Using
% forward propagation and pre-trained thetas.
% -------------------------------------------
X1 = [ones(m, 1) X]; % Add bias column
X2 = [ones(m, 1) sigmoid(X1 * Theta1')];
X3 = sigmoid(X2 * Theta2');

% -------------------------------------------
%Convert label vectors to matrices
% -------------------------------------------
y_matrix = eye(num_labels)(y,:);

% -------------------------------------------
% Calculate regularization term
% -------------------------------------------
rtheta1      = Theta1;
rtheta2      = Theta2;
rtheta1(:,1) = 0;
rtheta2(:,1) = 0;
rtheta1_sq   = (rtheta1 .^ 2);
rtheta2_sq   = (rtheta2 .^ 2);
reg_J        = (lambda/(2*m)) * (sum(rtheta1_sq(:)) + sum(rtheta2_sq(:)));

% -------------------------------------------
% Calculate cost using forward propagation
% -------------------------------------------
J_matrix = (-1/m) * ((y_matrix .* log(X3)) + ((1-y_matrix) .* log(1-X3)));
J        = sum(J_matrix(:)) + reg_J;

% -------------------------------------------
% Backpropagation gardients
% -------------------------------------------
d3 = X3 - y_matrix;

%Excluding the first column of Theta2 while calculating delta is
%because the hidden layer bias unit has no connection to the input
%layer - so we do not use back-propagation for it.

d2 = (d3 * rtheta2(:,2:end)) .* sigmoidGradient(X1 * Theta1');

delta1 = d2' * X1;
delta2 = d3' * X2;

Theta1_grad = 1/m * (delta1 + (lambda * rtheta1));
Theta2_grad = 1/m * (delta2 + (lambda * rtheta2));

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
