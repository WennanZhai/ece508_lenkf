function mask = gen_gc_mask_1D(N, corr_l)

if (corr_l == 0)
  mask = speye(N);
else
  K = 0:2*corr_l-1;
  d = gc(K, corr_l)';
  
  J = find(abs(d) < 1e-6);
  d(J) = 0;
  
  d = [fliplr(d(2:end)), d];
  D = ones(N,1)*d;
  
  I = -2*corr_l+1:2*corr_l-1;
  
  mask = spdiags(D, I, N, N);
end
