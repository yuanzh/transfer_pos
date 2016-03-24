% CF_EVAL.M [#]
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% THE SCOPE OF THIS CODE:
% This Matlab(R) function evaluates the Brockett cost function: 
% J=trace(W'*S*W*N), where W is an n-by-n unitary matrix,
% S is an arbitrary Hermitian n-by-n matrix,
% and N is a diagonal matrix whose diagonal elements are 1,...,n.
%
% The evaluation is done at a point W=Wk on the Lie group of n-times-n 
% unitary matrices U(n).
%
% This script [#] is entirely specific to the Brockett cost function and 
% obviously needs to be completely rewritten for other cost functions.
% This is the file where to write the corresponding code.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% USAGE and OPTIONS:
% J = cf_eval(Wk)
% 
% INPUT:
% Wk = n-by-n unitary matrix 
% 
% GLOBAL VARIABLES (set in main_code.m)
% S = n-by-n Hermitian matrix in the Brockett criterion 
% N = n-by-n diagonal matrix in the Brockett criterion 
%
% OUTPUT:
% J = the Brockett cost function evaluated at Wk
% for details, see eq. (17) in [1] 
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

function J = cf_eval(Wk)
%global S N; % [#] the cost function-specific variables are global
%J=real(trace(Wk'*S*Wk*N)); % [#] the Brockett cost function value at Wk
% it is real-valued, but real part is taken just for numerical stability
global X Y;
diff = X * Wk - Y;
J = sum(sum(diff .* diff));
end





