% GEOD_SEARCH_ARMIJO.M
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% THE SCOPE OF THIS CODE:
% This Matlab(R) function implements the Armijo line search method used in
% [2] (see Table 2) to find a suitable step size along geodesics on the
% unitary Lie group U(n), such that certain smooth cost function is 
% optimized (minimized or maximized) along that geodedic.
% (for more details, see E. Polak, Optimization: Algorithms and Consistent 
% Approximations. New York: Springer-Verlag, 1997.)
%
% The geodesic emanates from a point Wk on U(n) in the tangent direction 
% of Hk*Wk at point Wk.
% The search direction is represented by the skew-Hermitian matrix Hk that
% lies in the Lie algebra, and is related to Hk*Wk by right translation.
%
% This code follows the steps given in Table 2 in [2]. 
% Notation of variables might slightly differ from the one in [2].
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% USAGE and OPTIONS:
% mu_opt_armijo=geod_search_armijo(Wk,Gk,Hk,opt)
% 
% INPUT:
% Wk = n-by-n unitary matrix representing the current point on U(n)
% Gk = n-by-n skew-Hermitian matrix corresponding to the positive/negative
%      Riemannian gradient direction Hk*Wk at Wk
% Hk = n-by-n skew-Hermitian matrix corresponding to the search direction Hk*Wk
% N_poly = the order of the approximation polynomial
% opt = parameter that decides whether the function should be maximized or minimized
% along the geodesic emanating from Wk in the direction Hk*Wk
% 
% OUTPUT:
% mu_opt_armijo = the optimum step size as the first minimum of the cost function
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


function mu_opt_armijo=geod_search_armijo(Wk,Gk,Hk,opt)

% test whether the size of the initial input matrices Wk, Gk, Hk are correct or not
if size(Wk,1)~=size(Wk,2) || size(Gk,1)~=size(Gk,2) || size(Hk,1)~=size(Hk,2) || size(Wk,1)~=size(Gk,1) || size(Gk,1)~=size(Hk,1)
    error('ERROR (GEOD_SEARCH_ARMIJO.M): input matrices Wk, Hk must be square matrices of the same size')
end

mu=1; % step size is initialized to 1 

% take care of the minimization/maximization option by changing the 
% direction of motion along the geodesic (the sign of the step size) 
% For details, see Remark at page 1139 in [2]
switch lower(opt)
    case 'min' % move towards a descent direction - minimization is performed
        R1=expm(-mu*skew(Hk));   % basic initial rotation matrix
        R2=R1*R1; % double step size by squaring (no expm needed)
        % test both conditions: keep the step size constant? This may save calculations
            if (cf_eval(Wk) - cf_eval(R2*Wk) >= mu*innerprod(Gk,Hk)) && (cf_eval(Wk) - cf_eval(R1*Wk) < (1/2)*mu*innerprod(Gk,Hk))
                mu=mu;  %fprintf('\n condition 0 \n'); % do nothing
            else
        % test first condition: doubling the step size?
                while cf_eval(Wk) - cf_eval(R2*Wk) >= mu*innerprod(Gk,Hk)  %&& abs(mu) < 2^8   
                    % upper bound for step size - this is unnecessary for SD, but for CG, Wolfe-Powell conditions may be violated
                    mu=2*mu;    %fprintf('\n condition 1 \n');  
                    R1=R2;  R2=R1*R1; % double step size by squaring (no expm needed)
                end
        % test second condition: halving the step size?
                while cf_eval(Wk) - cf_eval(R1*Wk) < (1/2)*mu*innerprod(Gk,Hk)   %&& abs(mu) > eps % lower bound for step size - avoid small number issues
                    mu=(1/2)*mu; %fprintf('\n condition 2 \n');
                    R1=expm(-mu*skew(Hk)); % expm is needed to halve the step size
                end
            end
    case 'max' % move towards an ascent direction - maximization is performed (see Remark at page 1139 in [2])
        R1=expm(+mu*skew(Hk));   % basic initial rotation matrix
        R2=R1*R1; % double step size by squaring (no expm needed)
        % test both conditions: keep the step size constant? This may save calculations
            if (cf_eval(Wk) - cf_eval(R2*Wk) <= -mu*innerprod(Gk,Hk)) && ((cf_eval(Wk) - cf_eval(R1*Wk)) > (1/2)*mu*innerprod(Gk,Hk))
                mu=1*mu; %fprintf('\n condition 0, keep the step size constant \n'); % do nothing
            else
    % test first condition: doubling the step size?
                while (cf_eval(Wk) - cf_eval(R2*Wk) <= -mu*innerprod(Gk,Hk))  %&& abs(mu) < 2^8
                    % upper bound for step size - this is unnecessary for SD, but for CG, Wolfe-Powell conditions may be violated
                    mu=2*mu;    %fprintf('\n condition 1, double the step size \n');  
                    R1=R2;  R2=R1*R1; % double the step size by squaring the rotation
                end
    % test second condition: halving the step size?
                while (cf_eval(Wk) - cf_eval(R1*Wk) > -(1/2)*mu*innerprod(Gk,Hk))   %&& abs(mu) > eps % lower bound for step size - avoid small number issues
                    mu=(1/2)*mu; %fprintf('\n condition 2, halve the step size \n');
                    R1=expm(+mu*skew(Hk));
                end
            end
    otherwise
         error('ERROR (GEOD_SEARCH_ARMIJO.M): Bad option for parameter OPT. Valid strings are: MIN, MAX')
end
% the final Armijo step size
mu_opt_armijo=mu;
