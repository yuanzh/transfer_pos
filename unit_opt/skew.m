% SKEW.M
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% THE SCOPE OF THIS CODE: [#]
% This Matlab(R) function is only needed to improve numerical stability
% of algorithms that rely on skew-Hermitian matrices.
% This code enforces exact skew-Hermitian property of a matrix.
% It also works for real-valued skew-symmetric matrices.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% USAGE and OPTIONS:
% S_out=skew(S_in)
% 
% INPUT:
% S_in  = n-by-n approximately skew-Hermitian (or skew-symmetric) matrix
%
% OUTPUT:
% S_out = n-by-n exact skew-Hermitian (or skew-symmetric) matrix
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

function S_out=skew(S_in)
if size(S_in,1)~=size(S_in,2) % check is matrix sizes are correct
    error('ERROR (SKEW.M): input matrix S_in must be square')
else
    S_out=0.5*(S_in-S_in'); % the exact skew-Hermitian matrix
end

