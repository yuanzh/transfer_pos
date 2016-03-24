% GEOD_SEARCH_DFT.M
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% THE SCOPE OF THIS CODE:
% This Matlab(R) function implements step-by-step the line search method
% based on DFT approximation, given in Table 2, in [1].
% It finds an optimum step size along a geodesic on the unitary Lie group 
% U(n), such that certain smooth cost function is minimized/maximized along
% that geodedic.
%
% Optimization is performed by using a DFT approximation of the
% first-order derivative of the cost function along the geodesic emanating
% from a point Wk in the tangent direction of Hk*Wk at point Wk.
% Several minima/maxima along the geodesic emanating from Wk in the 
% direction Hk*Wk are found. They corresponds to the first-zero-crossing 
% of the first-order derivative of the cost function along geodesic.
% The best minimum/maximum in the DFT window is selected.
% The search direction is represented by the skew-Hermitian matrix Hk that
% lies in the Lie algebra, and is related to Hk*Wk by right translation.
%
% This code follows the steps given in Table 2 in [1].
% Notation of variables might slightly differ from the one in [1].
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% USAGE and OPTIONS:
% mu_opt_dft=geod_search_dft(Wk,Hk,NT,opt)
% 
% INPUT:
% Wk = n-by-n unitary matrix representing the current point on U(n)
% Hk = n-by-n skew-Hermitian matrix corresponding to the search direction Hk*Wk
% NT = the number of almost-periods of the cost function contained in the DFT window
% opt = parameter that decides whether the function should be maximized or minimized
% along the geodesic emanating from Wk in the direction Hk*Wk
% 
% OUTPUT:
% mu_opt_dft = the optimum step size as the best minimum of the cost function
% along the geodesic emanating from Wk in the direction Hk*Wk, withing the DFT
% window
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


function mu_opt_dft=geod_search_dft(Wk,Hk,NT,opt)

global q; % global variable declared in the main code main_code.m

% test whether the size of the initial input matrices Wk, Hk are correct or not
if size(Wk,1)~=size(Wk,2) || size(Hk,1)~=size(Hk,2) || size(Wk,1)~=size(Hk,1)
    error('ERROR (GEOD_SEARCH_DFT.M): input matrices Wk, Hk must be square matrices of the same size')
end

% make sure the number of almost-periods NT is a strictly positive natural number
if (NT-ceil(NT) ~= 0) || (NT < 1)
   error('ERROR (GEOD_SEARCH_DFT.M): Bad option for parameter NT. It must be a strictly positive natural number') 
end

% take care of the minimization/maximization option by changing the direction
% of motion along the geodesic (the sign of the step size) 
switch lower(opt)
    case 'min'
         sign_mu=-1; % move towards a descent direction
    case 'max'
         sign_mu=+1; % move towards an ascent direction 
    otherwise
        error('ERROR (GEOD_SEARCH_DFT.M): Bad option for parameter OPT. Valid strings: MIN, MAX')
end

% 1) Wk, Hk are given as input parameters to the function
% compute the eigenvalue of Hk with the highest magnitude
omega_max=max(abs(eig(Hk))); % can be replaced by faster operation norm(Hk) 

% 2) Determine the order of the cost function 
% q = order of the cost fct. is a global variable set in main_code.m

% 3) Determine the almost period T_mu
T_mu=2*pi/(q*omega_max); % one almost-period

% 4) Choose the sampling factor K
K=3; % number of samples/almost-period (3,4 or 5)
% number of periods NT for the DFT-based approximation is given as an input

% 5) Determine the length of the DFT interval and DFT length
T_dft=NT*T_mu; % NT almost-periods T_mu
N_dft=2*floor(K*NT/2)+1; % DFT size - ensure odd value (otherwise, the closest one is chosen)

% 6) Evaluate the rotation matrix at equi-spaced points
R_dft=expm(sign_mu*(T_dft/N_dft)*Hk); % the rotation - computed only once!
R_mu_dft=eye(size(Wk));% rotation is initialized to identity (mu=0)

% 7) By using the computed rotation matrices, evaluate the 1st-order
%    derivative of the cost function
J_dft=zeros(N_dft,1);  % initialization for prior memory allocation 
d1_dft=zeros(N_dft,1); % initialization for prior memory allocation 
for n_dft=1:N_dft 
    Dk=euclid_grad_eval(R_mu_dft*Wk); % Euclidean grad. at R_mu_dft(mu_dft)*Wk
    % Note: obviously, euclid_grad_eval.m [#] is cost function-specific
    d1_dft(n_dft,1)=-2*sign_mu*real(trace(Dk*Wk'*R_mu_dft'*Hk')); % 1st derivative
    J_dft(n_dft,1)=cf_eval(R_mu_dft*Wk); % the sampled cost function
    R_mu_dft=R_mu_dft*R_dft; % powers of the rotation matrix
end

% 8) Compute the Hann window
h_window=hanning(length(d1_dft)); % use built-in function

% 9) Compute the windowed derivative
d1_dft=d1_dft.*h_window; % windowing operation

% 10) Compute the Fourier coefficients 
coefs_dft=fftshift(fft(d1_dft))/N_dft; % DFT coefficients

% 11) Find roots of the dft polynomial that are close to the unit circle
poly_roots=roots(flipud(coefs_dft)); % all roots
circle_eps=10^(-2); % max distance from the unit circle
% find the roots close to the unit circle: exp(j*omega)
poly_roots=poly_roots(abs(abs(poly_roots)-1)<circle_eps);
log_poly_roots=log(poly_roots); % the exponent j*omega
imag_log_poly_roots=imag(log_poly_roots); % the imaginary part: omega

% 12) find the zero-crossing values corresponding to the roots
mu_roots=mod(imag_log_poly_roots/(2*pi/T_dft),T_dft); % avoid aliases
[sorted_mu,~]=sort(mu_roots,'ascend'); % ascending ordering
mu_opt_dft=sorted_mu(1:2:end); % take only the odd values
% in case no root is found or the derivative is small enough
if isempty(mu_opt_dft) || abs(d1_dft(1))<=1e-12
    mu_opt_dft=0; % step size is set to zero 
    % this actually belongs to step 11 in Table 2, in [1]), but
    % implementation is more straight-forward this way
else
    
% 13) Find the local optima of the cost fucntion along geodesic within the DFT range
mu_dft=linspace(0,NT*T_mu,N_dft+1).'; mu_dft=mu_dft(1:end-1); % the equi-spaced points
if opt == 'min' % minimization along geodesic
    [~,ind_Jdft_min]=min(J_dft); % the global minimum of the function evaluated at equi-spaced grid
    mu_dft_min=mu_dft(ind_Jdft_min); % find the value of mu corresponding to this minimum
    [~,ind_mu_min_min]=min(abs(mu_dft_min-mu_opt_dft)); % find the index of mu that is the closest to the equi-spaced grid
    mu_opt_dft=mu_opt_dft(ind_mu_min_min); % find the optimum value of along geodesic within the DFT range
else % opt == 'max' - maximization along geodesic
    [~,ind_Jdft_max]=max(J_dft); % the global maximum of the function evaluated at equi-spaced grid
    mu_dft_max=mu_dft(ind_Jdft_max); % find the value of mu corresponding to this maximum
    [~,ind_mu_max_max]=min(abs(mu_dft_max-mu_opt_dft)); % find the index of mu that is the closest to the equi-spaced grid
    mu_opt_dft=mu_opt_dft(ind_mu_max_max); % find the optimum value of along geodesic within the DFT range  
    end
end
