% GEOD_SEARCH_POLY.M
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% THE SCOPE OF THIS CODE:
% This Matlab(R) function implements step-by-step the line search method
% based on polynomial approximation, given in Table 1, in [1].
% It finds an optimum step size along a geodesic on the unitary Lie group 
% U(n), such that certain smooth cost function is minimized/maximized along
% that geodedic.
%
% Optimization is performed by using a polynomial approximation of the
% first-order derivative of the cost function along the geodesic emanating
% from a point Wk in the tangent direction of Hk*Wk at point Wk.
% The first minimum/maximum along the geodesic emanating from Wk in the 
% direction Hk*Wk is found. This corresponds to the first-zero-crossing 
% of the first-order derivative of the cost function along geodesic.
% The search direction is represented by the skew-Hermitian matrix Hk that
% lies in the Lie algebra, and is related to Hk*Wk by right translation.
%
% This code follows the steps given in Table 1 in [1]. 
% Notation of variables might slightly differ from the one in [1].
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% USAGE and OPTIONS:
% mu_opt_poly=geod_search_poly(Wk,Hk,N_poly,opt)
% 
% INPUT:
% Wk = n-by-n unitary matrix representing the current point on U(n)
% Hk = n-by-n skew-Hermitian matrix corresponding to the search direction Hk*Wk
% N_poly = the order of the approximation polynomial
% opt = parameter that decides whether the function should be maximized or minimized
% along the geodesic emanating from Wk in the direction Hk*Wk
% 
% OUTPUT:
% mu_opt_poly = the optimum step size as the first minimum of the cost function
% along the geodesic emanating from Wk in the direction Hk*Wk
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


function mu_opt_poly=geod_search_poly(Wk,Hk,N_poly,opt)

global q; % global variable declared in the main code main_code.m

% test whether the size of the initial input matrices Wk, Hk are correct or not
if size(Wk,1)~=size(Wk,2) || size(Hk,1)~=size(Hk,2) || size(Wk,1)~=size(Hk,1)
    error('ERROR (GEOD_SEARCH_POLY.M): input matrices Wk, Hk must be square matrices of the same size')
end

% test whether the W0 is a uniraty matrix and Hk is skew-Hermitian
if norm(Wk*Wk'-eye(size(Wk))) > 1e-5 || norm(Hk+Hk') > 1e-5 % generous threshold
    error('ERROR (GEOD_SEARCH_POLY.M): input matrices are wrong: Wk must be unitary and Hk must be skew-Hermitian')
end

% make sure the plynomial order is a natural number between 3 and 5
if (N_poly-ceil(N_poly) ~= 0) || (N_poly < 3) || (N_poly > 5)
   error('ERROR (GEOD_SEARCH_POLY.M): Bad option for parameter N_POLY. It must be either 3, 4, or 5') 
end

% take care of the minimization/maximization option by changing the direction
% of motion along the geodesic (the sign of the step size) 
switch lower(opt)
    case 'min'
         sign_mu=-1; % move towards a descent direction 
    case 'max'
         sign_mu=+1; % move towards an ascent direction 
    otherwise
        error('ERROR (GEOD_SEARCH_POLY.M): Bad option for parameter OPT. Valid strings: MIN, MAX')
end


% 1) Wk, Hk are given as input parameters to the function
%    compute the eigenvalue of Hk with the highest magnitude
omega_max=max(abs(eig(Hk))); % can be replaced by faster operation norm(Hk) 

% 2) Determine the order of the cost function 
%    q = order of the cost fct. is a global variable set in main_code.m

% 3) Determine the almost period T_mu
T_mu=2*pi/(q*omega_max); % one almost-period

% 4) Choose the order of the approximating polynomial
%    - given as input parameters to the function

% 5) Evaluate the rotation matrix at equi-spaced points
mu_step_poly=T_mu/N_poly; % the sampling step for mu
R_poly=expm(sign_mu*mu_step_poly*Hk); % the rotation - computed only once!

R_mu_poly=eye(size(Wk)); % rotation is initialized to identity (mu=0)
for n_poly=1:N_poly+1
    
% 6) By using the computed rotation matrices, evaluate the 1st-order
%    derivative of the cost function
    Dk=euclid_grad_eval(R_mu_poly*Wk); % Euclidean gradient 
    % Note: obviously, euclid_grad_eval.m [#] is cost function-specific
    d1_poly(n_poly,1)=-2*sign_mu*real(trace(Dk*Wk'*R_mu_poly'*Hk')); % 1st derivative
    R_mu_poly=R_mu_poly*R_poly; % powers of the rotation matrix
end

% 7) Compute the polynomial coefficients
C=[]; % initialization of the the Vandermonde matrix
mu_poly=linspace(0,T_mu,N_poly+1).'; % equi-spaced sampling
for n_poly=1:N_poly
C=[C mu_poly(2:end).^n_poly]; % the Vandermonde matrix
end
% the polynomial coefficients
a=[d1_poly(1); inv(C)*(d1_poly(2:end)-d1_poly(1))]; % LS solution 

% 8) Find the smallest real positive root of the approximationg polyomial
all_roots_d1=roots(flipud(a)); % all polynomial roots
index_real_roots_d1=find(abs(imag(all_roots_d1))<=eps); % find real roots
real_roots_d1=(all_roots_d1(index_real_roots_d1)); % the real-valued roots
index_positive_roots_d1=find(real_roots_d1>0); % find positive roots
mu_opt_poly=min(real_roots_d1(index_positive_roots_d1)); % first positive root
% in case no root is found or the derivative is small enough
if isempty(mu_opt_poly) || abs(d1_poly(1,1))<1e-12
    mu_opt_poly=0;  % step size is set to zero
end
