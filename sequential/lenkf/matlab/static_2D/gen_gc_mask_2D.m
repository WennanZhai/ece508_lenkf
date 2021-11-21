function mask = gen_gc_mask_2D(res, corr_l)

if (length(res) == 1)
  M = res;
  N = res;
elseif (length(res) == 2)
  M = res(1);
  N = res(2);
else
  error('length(res) must equal 1 or 2');
end
  
if (corr_l == 0)
  mask = speye(M*N, M*N);
else
  tau = ceil(2*corr_l);
  if (mod(tau, 2) == 0)
    tau = tau + 1;
  end
  
  B = zeros(tau);
  
  I = -(tau-1)/2:(tau-1)/2;
  for i=1:tau
    d = sqrt(I(i)^2 + I.^2);
    B(i, :) = gc(d, corr_l);
  end
  
  
  if (0)
    figure(2);
    imagesc(B);
    colorbar;
  end
  
  mask = sparse(M*N, M*N);
  
  center = (tau-1)/2;

  for i=1:N
    for j=1:M
      P = zeros(M,N);
      
      for k=(i-center):(i+center)
        for l=(j-center):(j+center)
          if (k >= 1 && k <= N && l >= 1 && l <= M)
            P(l, k) = B((j-l)+center+1, (i-k)+center+1);
          end
        end
      end

      if (0)
      figure(1);
      imagesc(P);
      disp(sprintf('i=%d  j=%d  row=%d', i, j, (i-1)*M + j));
      pause;
      end
    
      mask(((i-1)*M + j),:) = P(:)';
    end
  end
end