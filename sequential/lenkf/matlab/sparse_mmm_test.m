% sparse_mmm_test.m

rand('twister', 1);
randn('state', 1);

N = 64;
M = 32;
L1 = 4*N;
L2 = 4*floor(sqrt(N*M));

C_test = zeros(N, N);
C_test(ceil(rand(L1,1)*N*N)) = randn(L1,1);
C_test = C_test*C_test';

H_test = zeros(M, N);
H_test(ceil(rand(L2,1)*M*N)) = randn(L2,1);

P_HT_true = C_test*H_test';

[c,r_temp,v] = find(C_test');
C_v = v;
C_j = c - 1;
r = zeros(N+1, 1);
for i=1:N
  r(i+1) = length(find(r_temp == i));
end
C_r = cumsum(r);


[c,r_temp,v] = find(H_test');
H_v = v;
H_j = c - 1;
r = zeros(M+1, 1);
for i=1:M
  r(i+1) = length(find(r_temp == i));
end
H_r = cumsum(r);

P_HT = zeros(N,M);


for i=0:N-1
  index_C = 0;
  
  for n=0:M-1
    nnz_Ci = C_r((i+1)+1) - C_r((i)+1);
    
    index_H = H_r((n)+1);
    for j=0:nnz_Ci-1
      index_C = j + C_r((i)+1);
      if (C_j((index_C)+1) < i)
        continue;
      end
      
      while(H_j((index_H) + 1) < C_j((index_C)+1))
        index_H = index_H + 1;
        if (index_H >= H_r((n+1)+1))
          break;
        end
      end
      
      if (index_H >= H_r((n+1)+1))
        break;
      end
      
      
      if (H_j((index_H)+1) == C_j((index_C)+1))
        P_HT((i)+1, (n)+1) = P_HT((i)+1, (n)+1) ...
            + C_v((index_C)+1) * H_v((index_H) + 1);
      end
      
      if (H_j((index_H)+1) == i)
        for j2=j+1:nnz_Ci-1
          index_C2 = j2 + C_r((i)+1);
          i2 = C_j((index_C2)+1);
          P_HT((i2)+1, (n)+1) = P_HT((i2)+1, (n)+1) ...
              +  C_v((index_C2)+1) * H_v((index_H)+1);
        end
      end
    end
  end
end

norm(P_HT(:) - P_HT_true(:))