function [Q, Q_sqrt, Q_sqrt_rs22] = gen_Q(res, corr_l)

if (length(res) == 1)
  M = res;
  N = res;
elseif (length(res) == 2)
  M = res(1);
  N = res(2);
else
  error('length(res) must equal 1 or 2');
end

assert(mod(corr_l, 2) == 0);


Q = sparse(M*N, M*N);

c_corr_l = ceil(corr_l);
  
Q_sqrt_rs22 = zeros(c_corr_l);
[l,l] = size(Q_sqrt_rs22);

a = c_corr_l/(l-1);
b = -a - c_corr_l/2;

for i = 1:l
  for j = 1:l
    d = sqrt((a*i+b)^2 + (a*j+b)^2);
    
    if (d <= corr_l/2+1);
      Q_sqrt_rs22(i,j) = sqrt((corr_l/2+1)^2 - d^2) * (corr_l/2+1)^(-1);
    end
  end
end

Q_sqrt = convmtx2(Q_sqrt_rs22, M, N)';
Q = Q_sqrt*Q_sqrt';

%s = svd(Q);
%max_sv = s(1);
max_sv = svds(Q,1);
Q = Q/max_sv;
Q_sqrt = Q_sqrt/sqrt(max_sv);
Q_sqrt_rs22 = Q_sqrt_rs22/sqrt(max_sv);