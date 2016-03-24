function [W, obj] = unitary_project(Xl, Yl)

% DEFINE COST FUNCTION SPECIFIC VARIABLES AS GLOBAL [#]
global q X Y; % [#]
X = Xl;
Y = Yl;
%
% DEFINE THE OPTIMIZATION PARAMETERS
n=size(X, 2); % choose the size of n-byn unitary matrix W, set to n = 2, 3, ...
%
% choose a starting point for the alogorithms
W0=eye(n); % initialization at identity (can be any non-stationary point)
% [W0,~]=qr(randn(n)+j*randn(n)); % a random initial unitary matrix
%
% choose whether the cost function should be minimized mazimized, below:
% opt='min'; % optimization done: set to 'min' = minimization
opt='min'; % optimization done:  set to 'max' = maximization (try this too)
%
% Choose the number of iterations for the gradient algorithm
K_iter=100; % a finite number of iteration, set to K = 1, 2, ...

q=2; % [#] the degree of the Brockett function in coefficients of W (quadratic)
% (q is a strictly positive natural number)
% set the size of the n-times-n matrices S, W and N in the Brocket function
%S=randn(n)+j*randn(n); S=S*S'; % [#] generate some positive Hermitian matrix
%N=diag(1:n); %  % [#] diagonal matrix with the first n natural numbers
%X = rand(10, n);
%Y = rand(10, n);

% call the Riemannian optimization script for all combinations of gradient
% algorithms (SD/SA, CG-PR, CG-FR and line search methods (Armijo,
% polynomial and DFT approximation-based methids, respectively.
%
% SD + Armijo method
[W_final_sdsa_a,J_dB_sdsa_a,E_dB_sdsa_a,U_dB_sdsa_a]=riemann_grad_unit_opt(W0,'sdsa','a',opt,K_iter);
W = W_final_sdsa_a;
obj = J_dB_sdsa_a(K_iter, 1);
