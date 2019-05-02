function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

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
%
% Note: grad should have the same dimensions as theta
%

% the following two lines are not correct
% J = (1/m) * sum(-y * log(sigmoid(X*theta)) - (1-y) * log(1-sigmoid(X*theta)));
% grad = (1/m) * sum(sigmoid((X*theta) - y)*X); 

% ------------- Note and Explaination of my code ------------
% By Shawn
% The implementation of the code is better to use Vectorization techniques 
% mentioned in "Vectorization" in "Octave/Matlab Tutorial" in Week 2

% In sigmoid(), it's (transpose of theta) * (X) represents the equation:
% theta0 + theta1*x1 + theta2*x2 + theta3*x3 + ......

% the sum of y(i)*log(h(x(i))) = y' * log(sigmoid((theta')*X))
% so the cost function could be written as the following:

% J = (1/m) * ((-y)' * log(sigmoid(theta' * X)) - (1-y)' * log(1 - sigmoid(theta' * X)));
% grad = (1/m) * X' * (sigmoid(theta' * X) - y);

% The two lines of J and gradient are still not right (2019.4.26), currently I still cannot fully figure
% out all the details of the implementation of Cost Function in Octave, come back later.

% for a very good explaination of everything about the following two line of code, go to folder of programming_exercise3 in Week4, page 4 and 5 under chapter 1.3 Vectorizing Logistic Regression in ex3.pdf,
% ***** there's a very good explaination for everything about the implementation of CostFunction. it should help understand this if somehow forgot.
J = (1/m) * (y' * log(sigmoid(X*theta)) - (1-y)' * log(1 - sigmoid(X*theta)));
grad = (1/m) * X' * (sigmoid(X*theta) - y);


% =============================================================

end
