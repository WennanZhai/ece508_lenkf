function x_hat = sparse_cg(H, y, x0, max_it)
% Solves H'*H*x = H'*y using the conjugate gradient algorithm
% (algorithm 10.2.1 in Golub and van Loan, see also section 10.2.6).  

if (nargin == 3)
  max_it = inf;
end

[m,n] = size(H);

%CG_EPS = 1e-12;
CG_EPS = eps;
RECOMPUTE_RATE = floor(2*n);

if (nargin == 2)
  x0 = zeros(n,1);
end

x_hat = x0;

b = (y'*H)';

z = H*x0;
r = b - (z'*H)';

k = 0;
rho = r'*r;

THRESHOLD = CG_EPS*norm(b);

while (sqrt(rho) > THRESHOLD && k <= max_it)
  k = k + 1;
  
  if (0)
  if (mod(k, RECOMPUTE_RATE) == 1)
    z = H*x_hat;
    r = b - H_T*z;
    
    k = 1;
  end
  end

  if (mod(k,100) == 0)
    disp(sprintf('%d of %d', k, n))
  end
  
  if (k == 1)
    p = r;
  else
    beta = rho/rho_old;
    p = r + beta*p;
  end

  z = H*p;
  z = (z'*H)';
  p_T_A_p = p'*z;
  
  alpha = rho/p_T_A_p;
  x_hat = x_hat + alpha*p;
  r = r - alpha*z;
  rho_old = rho;
  rho = r'*r;
end

disp(sprintf('# iterations: %d', k));