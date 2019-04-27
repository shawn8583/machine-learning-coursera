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

% this part I used a for loop to generate a theta_reg that theta0 = 0 so the theta0
% will not be regularized in the last term of J
% that means theta_reg = [0; theta1; theta2; theta3; theta4; ......]
a = [theta(1,1)];

for i = 1 : (size(theta)(1,1) - 1)
    a = [a; 0];
end

theta_reg = theta - a;

% Now computes J and gradient, these are similiar from CostFunction.m
J = (1/m) * (y' * log(sigmoid(X*theta_reg)) - (1-y)' * log(1 - sigmoid(X*theta_reg))) + (lambda/(2*m)) * (theta_reg*(theta_reg'));
grad0 = (1/m) * X' * (sigmoid(X*theta - y));     % compute gradient value for theta0
grad_rest = (1/m) * X' * (sigmoid(X*theta_reg - y)) + (lambda/m)*theta_reg;       % compute gradient value for theta other than theta0

grad_rest(1,:) = [];     % delete the unit where theta0 was computed in the wrong way
grad_reg = [grad0; grad_rest];      % compose the values for j = 0 and for j >= 1


% =============================================================

end
