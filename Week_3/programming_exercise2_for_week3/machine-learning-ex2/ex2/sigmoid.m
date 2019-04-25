function g = sigmoid(z)
%SIGMOID Compute sigmoid function
%   g = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).

for m=1:size(z, 1)
    for n=1:size(z, 2)
        g(m, n) = 1/(1+exp(-z(m, n)));
    end
end


% =============================================================

end
