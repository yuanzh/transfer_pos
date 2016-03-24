% MAIN_CODE.M
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% THE SCOPE OF THIS CODE:
% This is the MAIN CODE for testing the numerical optimization algorithms 
% and the line search methods proposed in [1] and [2] for optimization 
% under unitary matrix constraint (see the references below).
% An arbitrary smooth cost function J(W) is optimized iteratively by using
% Riemannian gradient algorithms that operate on the Lie group of n-times-n 
% unitary matrices U(n). This is equivalent with finding an n-by-n unitary
% matrix W_final that yields the optimum (minimum or maximum) value of the  
% smooth function J. This is exactly what this code does. In general, local 
% optimization is achieved, but in some cases (depending on the nature of 
% the cost function), global optimization may occur. In many applications,
% local optimization is sufficient (see [1,2]).
%
%   Find an optimum unitary matrix W_final, such that smooth J(W) is 
%   minimized or maximized at W = W_final.
%
% ALL COMBINATIONS OF RIEMANNIAN GRADIENT ALGORITHMS AND 
% LINE SEARCH METHODS in [1,2] WILL BE TESTED:
%
% RIEMANNIAN OPTIMIZATION ALGORITHMS to be tested:
%  SD/SA: Steepest Descent/Ascent (see Table 1 in [2]), 
%  CG-PR: Conjugate Gradient (Polak-Ribiere formula) (see Table 3 in [1])
%  CG-FR: Conjugate Gradient (Fletcher-Reeves formula) (see Table 3 in [1])
%
% LINE SEARCH METHODS to be tested:
%   Armijo step method, (see Table 2 in [2]), 
%   polynomial approximation-based method (see Table 1 in [1])
%   DFT approximation-based method (see Table 2 in [1])
%
% All algorithms and methods are implemented step-by-step, as in the Tables
% given in [1,2]. 
% 
% In practical applications, just one of these algorithms (SD/SA, CG-PR/CG-FR)
% together with just one line search method (Armijo, polynomial of DFT-based)
% is sufficient to solve the problem at hand. The algorithm/method to be 
% used may be chosen based on experimental testing, the difference in 
% performance may depend on the cost function. In general, based on our 
% experience, first we recommend CG-PR with the polynomial-based line search 
% method, and second, SD/SA with the DFT-based line search method.
% 
% This code calls the function riemann_grad_unit_opt.m - for SD/SA, CG-PR, CG-FR
% that further calls the following functions:
%   geod_search_armijo.m  - for Armijo-based line search method
%   geod_search_poly.m - for polynomial-based line search method
%   geod_search_dft.m  - for DFT-based line search method
%
% Just as an example, the Brockett cost function: J=trace(W'*S*W*N) 
% is minimized/maximized w.r.t an n-times-n unitary matrix W, i.e., 
% W'*W=W*W'=eye(n), where S is an n-times-n positive Hermitian matrix,
% and N is a diagonal matrix whose diagonal elements are 1,...,n.
% The solution W of this optimization problem is known, i.e. it is given by
% the eigenvectors of S. Therefore, the matrix W'*S*W will be a diagonal
% matrix that contains the eigenvalues of S along the diagonal. 
% The eigenvectors can be obtained either by minimizing, or by maximizing
% the Brockett cost function, only the ordering along the diagonal of the
% diagonalized matrix W'*S*W will be different (increasing/decreasing order). 
%
% These codes can also be used to other smooth cost functions, 
% simply by changing the cost function-specific parameters/scripts. 
% In order to make this adaptation easier, the lines of code that are specific
% to the cost function are marked throughout the scripts with the sign [#].
% In particular, the changes that are required for optimizing another cost 
% functions are listed below.
%
% I.  In this script (main_code.m): Obviously, matrices that appear in the 
%     expression of the Brockett cost function (S, N) need to be replaced 
%     by other matrices that appear in the new cost function. Our goal was 
%     to make the Riemannian optimization scripts independent of the cost 
%     function. 
%     The script for gradient optimization (riemann_grad_unit_opt.m) and the
%     scripts that implement the three different line search methods
%     (geod_search_poly.m, geod_search_armijo.m, geod_search_dft.m)
%     are separated form the cost function-specific scripts (cf_eval.m,
%     euclid_grad_eval.m). The matrices that appear in the expression of
%     the cost function are declared as global variables in this main code
%     (see below). Also the order q of the cost function w.r.t. W needs to
%     be set accordingly (here: q=2 because the Brockett function is
%     quadratic in W). In order to set the right value of q for other 
%     functions, see step 2 in Tables 1 and 2 (and the corresponding 
%     comments in Section 3.1) in [1].
%
% II. Only two cost function-specific scripts needs to be changed for the 
%     particular cost function to be used. These are:
%     - cf_eval.m [#]: This needs to be completely rewritten, as it
%     evaluates a particular cost function (here the Brockett function) 
%     at a given point Wk on U(n).
%     - euclid_grad_eval.m [#]: This also needs to be completely rewritten,
%     as it evaluates the Euclidean gradient of a particular cost function
%     (here the Brockett function) at a given point Wk on U(n). 
%     Fortunately, these scripts have in general just few lines of code and
%     the changes may be done straight-forwardly.
%
% III. The code riemann_grad_unit_opt.m also calls the function diag_crit_eval.m 
%     that computes the diagonality criterion E_dB (see eq. (18) in [1], 
%     first expression). The diagonality criterion may be useful for other 
%     cost functions as well (e.g. JADE cost function [1,2]). Otherwise,
%     the call of this script may be removed from riemann_grad_unit_opt.m by 
%     commenting the following line:
%     E(k,1)=diag_crit_eval(Wk); % diagonality criterion (see eq. (17) in [1])
%     as well as the line at the end of the code:
%     E_dB=10*log10(abs(E)); % the diagonality criterion in [dB]
%     This also requires the riemann_grad_unit_opt.m function to be called by 
%     skipping the variable E_dB, as below:
%     [W_final,J_dB,~,U_dB]=riemann_grad_unit_opt(W0,grad_method,geod_search_method,opt,K_iter)
%     In this case, the Matlab plot lines below, that refer to E_dB should
%     also be commented.
%
% No changes are required in the codes corresponding to Rimenannian 
% optimization gradient algorithms, neither in the line search methods. 
% This makes these codes easily adaptable to other unitary optimization 
% problems, just by changing the two cost functions specific scripts 
% mentioned at item II above.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% USAGE and OPTIONS:
% 
% DEFINITION OF GENERAL VARIABLES IN THE MAIN CODE (this code):
% The codes runs all combinations of Riemannian gradient algorithms (SD/SA,
% CG-PR, CG-FR) with the line search (Armijo, polynomial approximation,
% and DFT approximation-based methods) in [1,2] in order to perform 
% numerical optimization under unitary matrix constraint. 
%
% n = the dimension of the n-by-n unitary matrix W
% W0 = n-by-n unitary matrix for initialization (typically identity matrix)
% opt = parameter that decides whether the function should be maximized 
%     or minimized. This is done by setting variable "opt" to one of the
%     following strings: 
%       'min': the function will be minimized
%       'max': the function will be maximized
% K_iter = number of iterations for all algorithms
% 
% Algorithm choice:
% 1. The gradient method is chosen by setting the variable "grad_method"
%     when calling riemann_grad_unit_opt.m to one of the following strings: 
%       'sdsa': SD/SA algorithm (see Tables 1 in [2])
%       'cgpr': CG algorithm (Polak-Ribierre) (see Table 3 in [1])
%       'cgfr': CG algorithm (Fletcher-Reeves) 
% 2. The geodesic (line) search method is chosen by setting the variable 
%    "geod_search_method" when calling riemann_grad_unit_opt.m to one of the 
%    following strings: 
%       'a': Armijo method (see Table 2 in [2])
%       'p': polynomial approximation method (see Table 1 in [1])
%       'f': Fourier approximation method - DFT (see Table 2 in [1])
%
% DEFINITION OF COST FUNCTION-SPECIFIC VARIABLES IN THE MAIN CODE (this code):
% q = the order of the cost function J(W) in the coefficient of W
% S = random Hermitian matrix that appears in the Brockett criterion [#]
% N = diagonal matrix with diagonal elements 1,...n that also appears in 
%     the  Brockett criterion [#]
%
% OUTCOMES OF THE MAIN CODE (this code):
% I. The output variables are:
%   W_final = the final solution W optimizing the Brockett criterion
%   J_dB = the Brockett criterion  [dB] (see eq. (17) in [1])
%   E_dB = diagonality criterion [dB] (see eq. (18) in [1], first expression)
%   U_dB = unitarity criterion [dB] (see eq. (18) in [1]), second expression)
%     The names of the variables W_final, J_dB, E_dB, U_dB are suffixed by:
%       _sdsa for SD/SA algorithms
%       _cgpr for CG-PR algorithm
%       _cgfr for CG-PR algorithm
%     and further suffixed by:
%       _a for Armijo line search method
%       _p for polynomial approximation-based line search method
%       _f for DFT approximation-based line search method
% II. The output figures: 3-by-3 matrix plot with 
% rows   :  Brocket criterion / Diagonality criterion / Unitarity criterion
% columns:  SD / CG-PR / CG-FR algorithms
% legends:  Armijo method / polynomial method / DFT method
% text   :  the type of optimization that was performed (minimization or 
%           maximization) is printed on the figures corresponding to the 
%           cost function.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% COPYRIGHT and TERMS OF USE:
%
% This work was done at Aalto University, Espoo, Finland, during 2005-2008 
% together with Jan Eriksson and Visa Koivunen who are greatly acknowledged.
% This code should ONLY be used for educational and scientific purposes 
% (e.g. to be compared to other algorithms), and in non-commercial scopes.
% 
% These codes comes for free as they are, and the author does not assume 
% any responsibility for their usage.
% The authors support reproducible research and open software and therefore,
% they require their credits to be given. 
% In case these codes are used, please cite the corresponding papers as
% follows.
%
% When using the Conjugate Gradient (CG) algorithm:
%
% [1] T. Abrudan, J. Eriksson, V. Koivunen,
% "Conjugate Gradient Algorithm for Optimization Under Unitary Matrix Constraint", 
% Signal Processing, vol. 89, no. 9, Sep. 2009, pp. 1704-1714.
% PDF: http://www.sciencedirect.com/science/article/pii/S0165168409000814
% CITATION: http://signal.hut.fi/sig-legacy/unitary_optimization/AbrEriKoi09SP.txt
%
% When using the Steepest Descent/Ascent (SD/SA) algorithms:
%
% [2] T. Abrudan, J. Eriksson, V. Koivunen;
% "Steepest Descent Algorithm for Optimization under Unitary Matrix Constraint",
% IEEE Transactions on Signal Processing, vol. 56, no. 3, Mar. 2008, pp. 1134-1147. 
% PDF: http://ieeexplore.ieee.org/iel5/78/4451275/04436033.pdf?tp=&arnumber=4436033&isnumber=4451275
% CITATION: http://signal.hut.fi/sig-legacy/unitary_optimization/AbrEriKoi08TSP.txt 
%
% When using the polynomial-based or the DFT-based line search methods, 
% please cite [1].
%
% The codes were written by Traian Abrudan (C) 2007 
% Comments, questions and suggestions may be sent to abrudant@gmail.com
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all; clc;
%
% DEFINE THE OPTIMIZATION PARAMETERS
n=10; % choose the size of n-byn unitary matrix W, set to n = 2, 3, ...
%
% choose a starting point for the alogorithms
W0=eye(n); % initialization at identity (can be any non-stationary point)
% [W0,~]=qr(randn(n)+j*randn(n)); % a random initial unitary matrix
%
% choose whether the cost function should be minimized mazimized, below:
% opt='min'; % optimization done: set to 'min' = minimization
 opt='max'; % optimization done:  set to 'max' = maximization (try this too)
%
% Choose the number of iterations for the gradient algorithm
K_iter=100; % a finite number of iteration, set to K = 1, 2, ...

% DEFINE COST FUNCTION SPECIFIC VARIABLES AS GLOBAL [#]
global q S N; % [#]
%
q=2; % [#] the degree of the Brockett function in coefficients of W (quadratic)
% (q is a strictly positive natural number)
% set the size of the n-times-n matrices S, W and N in the Brocket function
S=randn(n)+j*randn(n); S=S*S'; % [#] generate some positive Hermitian matrix
N=diag(1:n); %  % [#] diagonal matrix with the first n natural numbers


% call the Riemannian optimization script for all combinations of gradient
% algorithms (SD/SA, CG-PR, CG-FR and line search methods (Armijo,
% polynomial and DFT approximation-based methids, respectively.
%
% SD + Armijo method
[W_final_sdsa_a,J_dB_sdsa_a,E_dB_sdsa_a,U_dB_sdsa_a]=riemann_grad_unit_opt(W0,'sdsa','a',opt,K_iter);
% SD + polynomial method
[W_final_sdsa_p,J_dB_sdsa_p,E_dB_sdsa_p,U_dB_sdsa_p]=riemann_grad_unit_opt(W0,'sdsa','p',opt,K_iter);
% SD + DFT method
[W_final_sdsa_f,J_dB_sdsa_f,E_dB_sdsa_f,U_dB_sdsa_f]=riemann_grad_unit_opt(W0,'sdsa','f',opt,K_iter);
% CG-PR + Armijo method
[W_final_cgpr_a,J_dB_cgpr_a,E_dB_cgpr_a,U_dB_cgpr_a]=riemann_grad_unit_opt(W0,'cgpr','a',opt,K_iter);
% CG-PR + polynomial method
[W_final_cgpr_p,J_dB_cgpr_p,E_dB_cgpr_p,U_dB_cgpr_p]=riemann_grad_unit_opt(W0,'cgpr','p',opt,K_iter);
% CG-PR + DFT method
[W_final_cgpr_f,J_dB_cgpr_f,E_dB_cgpr_f,U_dB_cgpr_f]=riemann_grad_unit_opt(W0,'cgpr','f',opt,K_iter);
% CG-FR + Armijo method
[W_final_cgfr_a,J_dB_cgfr_a,E_dB_cgfr_a,U_dB_cgfr_a]=riemann_grad_unit_opt(W0,'cgfr','a',opt,K_iter);
% CG-FR + polynomial method
[W_final_cgfr_p,J_dB_cgfr_p,E_dB_cgfr_p,U_dB_cgfr_p]=riemann_grad_unit_opt(W0,'cgfr','p',opt,K_iter);
% CG-FR + DFT method
[W_final_cgfr_f,J_dB_cgfr_f,E_dB_cgfr_f,U_dB_cgfr_f]=riemann_grad_unit_opt(W0,'cgfr','f',opt,K_iter);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  PLOTS  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

figure(1); clf
set(1, 'Position', [6 63 1205 625])
%
subplot(3,3,1)
J_dB_all=[J_dB_sdsa_a; J_dB_sdsa_p; J_dB_sdsa_f; J_dB_cgpr_a; J_dB_cgpr_p; J_dB_cgpr_f; J_dB_cgfr_a; J_dB_cgfr_p; J_dB_cgfr_f]; 
plot((0:K_iter-1).',J_dB_sdsa_a,'Color',0.4*[0 0 1],'LineWidth',2); grid on; hold on;
plot((0:K_iter-1).',J_dB_sdsa_p,'Color',0.4*[1 1 1],'LineWidth',2); 
plot((0:K_iter-1).',J_dB_sdsa_f,'--','Color',[0 0 0],'LineWidth',2); 
axis([0 K_iter-1 min(real(J_dB_all)) max(real(J_dB_all))+0.1]);
ylabel(['Brockett cost function [dB]'], 'FontSize',12);
xlabel('iteration', 'FontSize',12);
title('SD/SA', 'FontSize',20)
hold off 
legend('Armijo','poly','DFT','Location','East');
text(ceil(0.05*K_iter), 0.5*(min(real(J_dB_all))+max(real(J_dB_all))), [upper(opt) 'IMIZATION'],'FontSize',16)
%
subplot(3,3,2)
plot((0:K_iter-1).',J_dB_cgpr_a,'Color',0.4*[0 0 1],'LineWidth',2); grid on; hold on;
plot((0:K_iter-1).',J_dB_cgpr_p,'Color',0.4*[1 1 1],'LineWidth',2);
plot((0:K_iter-1).',J_dB_cgpr_f,'--','Color',[0 0 0],'LineWidth',2);
axis([0 K_iter-1 min(real(J_dB_all)) max(real(J_dB_all))+0.1]);
xlabel('iteration', 'FontSize',12);
title('CG-PR', 'FontSize',20)
hold off 
legend('Armijo','poly','DFT','Location','East');
text(ceil(0.05*K_iter), 0.5*(min(real(J_dB_all))+max(real(J_dB_all))), [upper(opt) 'IMIZATION'],'FontSize',16)

%
subplot(3,3,3)
plot((0:K_iter-1).',J_dB_cgfr_a,'Color',0.4*[0 0 1],'LineWidth',2); grid on; hold on;
plot((0:K_iter-1).',J_dB_cgfr_p,'Color',0.4*[1 1 1],'LineWidth',2);
plot((0:K_iter-1).',J_dB_cgfr_f,'--','Color',[0 0 0],'LineWidth',2);
axis([0 K_iter-1 min(real(J_dB_all)) max(real(J_dB_all))+0.1]);
xlabel('iteration', 'FontSize',12);
title('CG-FR', 'FontSize',20)
hold off 
legend('Armijo','poly','DFT','Location','East');
text(ceil(0.05*K_iter), 0.5*(min(real(J_dB_all))+max(real(J_dB_all))), [upper(opt) 'IMIZATION'],'FontSize',16)

%
subplot(3,3,4);
E_dB_all=[E_dB_sdsa_a; E_dB_sdsa_p; E_dB_sdsa_f; E_dB_cgpr_a; E_dB_cgpr_p; E_dB_cgpr_f; E_dB_cgfr_a; E_dB_cgfr_p; E_dB_cgfr_f]; 
plot((0:K_iter-1).',E_dB_sdsa_a,'Color',0.4*[0 0 1],'LineWidth',1); grid on; hold on;
plot((0:K_iter-1).',E_dB_sdsa_p,'Color',0.4*[1 1 1],'LineWidth',2); 
plot((0:K_iter-1).',E_dB_sdsa_f,'--','Color',[0 0 0],'LineWidth',2); 
axis([0 K_iter-1 min(E_dB_all) max(E_dB_all)+0.1]);
ylabel(['Diagonality criterion [dB]'], 'FontSize',12);
xlabel('iteration', 'FontSize',12);
hold off 
legend('Armijo','poly','DFT',3);
%
subplot(3,3,5)
plot((0:K_iter-1).',E_dB_cgpr_a,'Color',0.4*[0 0 1],'LineWidth',1); grid on; hold on;
plot((0:K_iter-1).',E_dB_cgpr_p,'Color',0.4*[1 1 1],'LineWidth',2);
plot((0:K_iter-1).',E_dB_cgpr_f,'--','Color',[0 0 0],'LineWidth',2);
axis([0 K_iter-1 min(E_dB_all) max(E_dB_all)+0.1]);
xlabel('iteration', 'FontSize',12);
hold off 
legend('Armijo','poly','DFT',3);
%
subplot(3,3,6)
plot((0:K_iter-1).',E_dB_cgfr_a,'Color',0.4*[0 0 1],'LineWidth',1); grid on; hold on;
plot((0:K_iter-1).',E_dB_cgfr_p,'Color',0.4*[1 1 1],'LineWidth',2);
plot((0:K_iter-1).',E_dB_cgfr_f,'--','Color',[0 0 0],'LineWidth',2);
axis([0 K_iter-1 min(E_dB_all) max(E_dB_all)+0.1]);
xlabel('iteration', 'FontSize',12);
hold off 
legend('Armijo','poly','DFT',3);
%
xlabel('iteration', 'FontSize',12);
hold off
legend('Armijo','poly','DFT',3);

subplot(3,3,7)
plot((0:K_iter-1).',U_dB_sdsa_a,'Color',0.4*[0 0 1],'LineWidth',1); grid on; hold on;
plot((0:K_iter-1).',U_dB_sdsa_p,'Color',0.4*[1 1 1],'LineWidth',2); 
plot((0:K_iter-1).',U_dB_sdsa_f,'--','Color',[0 0 0],'LineWidth',2); 
axis([0 K_iter-1 -360 0]);
ylabel(['Unitarity criterion [dB]'], 'FontSize',12);
xlabel('iteration', 'FontSize',12);
hold off 
legend('Armijo','poly','DFT',2);
%
subplot(3,3,8)
plot((0:K_iter-1).',U_dB_cgpr_a,'Color',0.4*[0 0 1],'LineWidth',1);  grid on; hold on;
plot((0:K_iter-1).',U_dB_cgpr_p,'Color',0.4*[1 1 1],'LineWidth',2);
plot((0:K_iter-1).',U_dB_cgpr_f,'--','Color',[0 0 0],'LineWidth',2);
axis([0 K_iter-1 -360 0]);
xlabel('iteration', 'FontSize',12);
hold off 
legend('Armijo','poly','DFT',2);
%
subplot(3,3,9)
plot((0:K_iter-1).',U_dB_cgfr_a,'Color',0.4*[0 0 1],'LineWidth',1); grid on; hold on;
plot((0:K_iter-1).',U_dB_cgfr_p,'Color',0.4*[1 1 1],'LineWidth',2);
plot((0:K_iter-1).',U_dB_cgfr_f,'--','Color',[0 0 0],'LineWidth',2);
axis([0 K_iter-1 -360 0]);
xlabel('iteration', 'FontSize',12);
hold off
legend('Armijo','poly','DFT',2);
%






