% RIEMANN_GRAD_UNIT.OPT.M
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% THE SCOPE OF THIS CODE:
% This Matlab(R) function minimizes/maximizes a smooth cost function: 
% J(W), w.r.t an n-times-n unitary matrix W: W'*W=W*W'=eye(n),
%
% Optimization is performed by using Riemannian gradient algorithms such as
%  - steepest descent/ascent (SD/SA) [2]) 
%  - conjugate gradient (CG) [1]
% They operate on the Lie group of n-times-n unitary matrices U(n).
%
% Three line search methods may be used together with SD and CG:
%  - Armijo step method
%  - a polynomial approximation-based method [1]
%  - a Discrete Fourier Transform (DFT) approximation-based method [1]
% 
% This script is general enough to optimize any smooth cost functions.
% It is a step-by-step implementation of the pseudo-codes given in Table 3 
% in [1] for CG algorithms. For SD/SA with Armijo step, the code follows 
% steps in Table 2 in [2]. General SD/SA from Table 1 in [1] may be 
% obtained by setting the CG parameter gamma_k to zero in Table 3 in [1]. 
% Notation of variables might slightly differ from the one in [1,2].
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% USAGE and OPTIONS:
% [W_final,J_dB,E_dB,U_dB]=riemann_grad_unit_opt(W0,grad_method,geod_search_method,opt,K_iter)
% 
% INPUT:
% W0 = the initial starting point (typically, the identity matrix,
%    but it can be any arbitrary unitary matrix).
% Algorithm choice:
% 1. The gradient method is chosen by setting the variable 
% grad_method to one of the following strings: 
%     'sdsa': SD algorithm (see Table 1 in [2])
%     'cgpr': CG algorithm (Polak-Ribiere) (see Table 3 in [1])
%     'cgfr': CG algorithm (Fletcher-Reeves) 
% 2. The geodesic (line) search method is chosen by setting the variable 
% geod_search_method to one of the following strings: 
%     'a': Armijo method (see Table 2 in [2])
%     'p': polynomial approximation method (see Table 1 in [1])
%     'f': Fourier approximation method - DFT (see Table 2 in [1])
% opt = parameter that decides whether the function should be maximized or minimized
% This is done by setting variable "opt" to one of the following strings: 
%     'min': the function will be minimized
%     'max': the function will be maximized
% OUTPUT:
% The output variables are:
%      W_final = the final solution W of the unitary optimization
%      J_dB = the value of the cost function  [dB] (see eq. (18) in [1])
%      E_dB = a diagonality criterion [dB] (see eq. (17) in [1])
%      U_dB = a unitarity criterion [dB] (see eq. (22) in [2])
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


function [W_final,J_dB,E_dB,U_dB]=riemann_grad_unit_opt(W0,grad_method,geod_search_method,opt,K_iter)

% test whether the size of the initial input matrix W0 is correct or not
if size(W0,1)~=size(W0,2) % check is matrix sizes are correct
    error('ERROR (RIEMANN_GRAD_UNIT_OPT.M): input matrix W0 must be square')
else
    n=size(W0,1); % get the matrix size
end

% test whether the W0 is a unitary matrix
if norm(W0*W0'-eye(n)) > 1e-5 % a very generous threshold
    error('ERROR (RIEMANN_GRAD_UNIT_OPT.M): input matrix W0 must be unitary')
end

% take care of the minimization/maximization option by changing the direction
% on motion along the geodesic (the sign of the step size) 
switch lower(opt)
    case 'min'
         sign_mu=-1; % move towards a descent direction (negative gradient)       
    case 'max'
         sign_mu=+1; % move towards an ascent direction (positive gradient)     
    otherwise
        error('ERROR (RIEMANN_GRAD_UNIT_OPT.M): Bad option for parameter OPT. Valid strings are: MIN, MAX')
end

% Step-by-step implementation of the pseudo-code given in Table 3 in [1].

% 1) initialization of Wk, that is the variable to be optimized, at iteration k
Wk=W0; % starting point given as an input parameter

% initialize cost function / diagonality criterion / unitarity criterion 
J=zeros(K_iter,1); E=zeros(K_iter,1); U=zeros(K_iter,1); % initializations
for k=1:K_iter % actually the index goes from zero, but due to Matlab 
               % index convention, here it needs to start from one
    % computing the figures at every step for performance evaluation
    J(k,1)=cf_eval(Wk); % cost function values (see eq. (17) in [1])
    E(k,1)=diag_crit_eval(Wk); % diagonality criterion (see eq. (18-1) in [1])
    U(k,1)=unit_crit_eval(Wk); % unitarity criterion (see eq. (18-2) in [1])

% 2) compute the Euclidean gradient, the Riemannian gradient and search 
%    direction at Wk translated to identity
    if mod(k-1,n^2)==0 % periodically reset the search direction 
        % (remeber that index value equal to one in this code corresponds 
        % to index value = zero in the algorithm given in Table 3 in [1].)
        GEk=euclid_grad_eval(Wk); % compute the Euclidean gradient at Wk
        % Note: obviously, euclid_grad_eval.m [#] is cost function-specific
        Gk=GEk*Wk'-Wk*GEk'; % compute the Riemannian gradient at Wk 
                            % translated to the group identity
        Hk=Gk; % periodically reset the search direction to the gradient Gk
    else
        Gk=Gkplus1; % Gk="the old Gkplus1"
        Hk=Hkplus1; % Hk="the old Hkplus1"
    end
    
% 3) no STOP condition is imposed here, the algorithm will run up to the 
%    maximum number of iterations "K_iter" that is given as an input

% 4) the chosen geodesic (line) search method
    switch lower(geod_search_method)
      %--------------------------------------------------------------------
      % Armijo line search method - see Table 2 in [2]
      % (see the corresponding script geod_search_armijo.m)
      case 'a'
        mu=geod_search_armijo(Wk,Gk,Hk,opt); % see the corresponding function geod_search_armijo.m
        R1=expm(sign_mu*mu*skew(Hk)); % skew operator improves the 
                     % numerical properties (maintains skew-symmetry of Hk)
                     % and therefore the unitary propoerty of R1
	  case 'p'
      %--------------------------------------------------------------------
      % the proposed polynomial approximation-based line search method
      % see Table 1 in [1] (see the corresponding script geod_search_poly.m)
	    N_poly=5; % approximation order for the polynomial-based line search method
	    mu=geod_search_poly(Wk,Hk,N_poly,opt); % see the corresponding function geod_search_poly.m
	    R1=expm(sign_mu*mu*skew(Hk)); % skew operator improves the 
                     % numerical properties (maintains skew-symmetry of Hk)
                     % and therefore the unitary propoerty of R1
      %--------------------------------------------------------------------
      % the proposed DFT approximation-based line search method
      % see Table 2 in [1] (see the corresponding script geod_search_dft.m)
	  case 'f'
	    NT=5; % Number of almost periods for the DFT-based line search method
	    mu=geod_search_dft(Wk,Hk,NT,opt); % see the corresponding function geod_search_dft.m
	    R1=expm(sign_mu*mu*skew(Hk)); % skew operator improves the 
                     % numerical properties (maintains skew-symmetry of Hk)
                     % and therefore the unitary propoerty of R1
      otherwise
        error('ERROR (RIEMANN_GRAD_UNIT_OPT.M): Bad option for parameter GEOD_SEARCH_METHOD. Valid strings are: a, p, f') 
          
            
    end % end of case sequence


% 5) the update 
Wkplus1=R1*Wk; % rotation R1=expm(sign_mu*mu*skew(Hk)) was computed above

% 6) compute the new Euclidean gradient at Wkplus1: GEkplus1
GEkplus1=euclid_grad_eval(Wkplus1); % 
%    compute the new Riemannian gradient at Wkplus1 translated to identity: Gkplus1
Gkplus1=GEkplus1*Wkplus1'-Wkplus1*GEkplus1';
%    the factor gamma_k:
%    Note: by setting gamma_k_k to zero we get a SD/SA algorithm instead of CG
        if strcmpi(grad_method,'sdsa')
            % generic SD/SA algorithms in Table 1 in [2] is obtained by
            % setting the parameter gamma_k to zero, that is equivalent to
            % setting the direction to the steepest descent/ascent 
            gamma_k=0; % => steepest descent/ascent instead of conjugate gradient
        elseif strcmpi(grad_method,'cgpr')
            % Polak-Ribiere formula
            gamma_k=innerprod(Gkplus1-Gk, Gkplus1)/innerprod(Gk,Gk); % using a simpler parallel transport corresponding to the Cartan-Schouten (+) connection
            % gamma_k=innerprod(Gkplus1-expm(0.5*mu*skew(Hk))*Gk*expm(0.5*mu*skew(Hk)),Gkplus1)/innerprod(Gk,Gk); % using the parallel transport of the Levi-Civita connection
            % gamma_k=max(gamma_k,0); % PR+ "Polak-Ribiere plus" formula
                                      % guarantees right search direction
        elseif strcmpi(grad_method,'cgfr')
            % Fletcher-Reeves formula
            gamma_k=innerprod(Gkplus1, Gkplus1)/innerprod(Gk,Gk);
        else
            error('ERROR (RIEMANN_GRAD_UNIT_OPT.M): Bad option for parameter GRAD_METHOD.  Valid strings: SDSA, CGPR, CGFR)')
            return
        end
%    compute the new search direction at Wkplus1: Hkplus1
Hkplus1=Gkplus1+gamma_k*Hk; 

% 7) test is the search direction is wrong 
	if innerprod(Gkplus1,Hkplus1) < 0 % this happens quite seldom
        % only PR+ formula above method guarantees that this does not happen
        Hkplus1=Gkplus1; % if wrong, reset to the gradinent direction
        % this message should not be seen, in general
        disp('Note: (RIEMANN_GRAD_UNIT_OPT.M): CG search direction was reset to SD/SA direction once (see riemnann_grad_opt.m)') 
    end
    
% 8) Update Wk    
Wk=Wkplus1; % k:=k+1, i.e., the new Wk is the old Wkplus1
end

% 0utput variables
W_final=Wk;
J_dB=10*log10(J); % the cost function values in [dB]
E_dB=10*log10(E); % the diagonality criterion in [dB]
U_dB=10*log10(U); % the unitarity criterion in [dB]

