% UNIT_CRIT_EVAL.M
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% THE SCOPE OF THIS CODE:
% This Matlab(R) function evaluates the unitarity criterion in [1] 
% (see eq. (18), the second expression)
% It measures the departure of Wk from the unitary property
% i.e., the departure from diagonal property of the product Wk'*Wk
% The evaluation is done at a point W=Wk on the Lie group of n-times-n 
% unitary matrices U(n). This code is needed to check the numerical
% stability of the unitary optimization, i.e., to check how much the
% solution departed from the unitary property.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% USAGE and OPTIONS:
% U = unit_crit_eval(Wk)
% 
% INPUT:
% Wk = n-by-n unitary matrix 
% 
% GLOBAL VARIABLES (set in main_code.m)
% S = n-by-n Hermitian matrix in the Brockett criterion 
%
% OUTPUT:
% U = the diagonality criterion, evaluated at Wk
% for details, see eq. (18), the second expression in [1] 
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

function U = unit_crit_eval(Wk)
% test whether the size of the initial input matrices Wk, Hk are correct or not
if size(Wk,1)~=size(Wk,2)
    error('ERROR (UNIT_CRIT.M): input matrix Wkmust be square')
end
n = size(Wk,1); % get the matrix size (assumed to be square)
% the unitarity criterion value at Wk
U=norm(Wk'*Wk-eye(n),'fro')^2; % departure of Wk from the unitary property
end
