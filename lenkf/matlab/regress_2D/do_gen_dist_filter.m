% do_gen_dist_filter.m

m_filt = [corr_l+1, corr_l+1];

k_filt = m_filt + n - 1;

f_wrap = @(x) f(x, (corr_l+1)/2);

a = gen_dist_filter(m_filt, f_wrap);

if (rank == 1)
  a_zp = zeros(1, k_filt);
else
  a_zp = zeros(k_filt);
end

I = cell(1, rank);
for i=1:rank
  I{i} = 1:m_filt(i);
end

a_zp(I{1:end}) = a;

if (rank == 1)
  a_zp = circshift(a_zp, [1, -(m_filt-1)]);
else
  a_zp = circshift(a_zp, -(m_filt-1));
end

A = convmtxn(a, n);

% Approximate
%Z = fftn(convn(full(a), full(a)), k_filt);
%gamma = 1/sqrt(Z(1));

% Exact
Z = A'*A;
gamma = 1/sqrt(svds(Z,1));

a = a * gamma;
a_zp = a_zp * gamma;
A = A * gamma;