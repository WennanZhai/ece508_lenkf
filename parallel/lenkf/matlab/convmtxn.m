function C = convmtxn(H, n, shape)

if (nargin == 2)
  shape = 'full';
end

rank = length(n);

s_H = zeros(1, rank);
for i=1:rank
  s_H(i) = size(H,i);
end

if (rank == 1)
  length(H) == length(n);
  
  X = zeros(n, 1);
  H = reshape(H, length(H), 1);
  M = length(H) + n - 1;
else
  X = zeros(n);

  rank = length(n);
  M = prod(s_H + n - 1);
end

N = prod(n);

if (strcmp(shape, 'full'))
  C = sparse(M, N);
elseif (strcmp(shape, 'same'))
  C = sparse(N, N);
elseif (strcmp(shape, 'valid'))
  a = n - s_H + 1;
  I = find(a < 0);
  a(I) = 0;
  
  C = sparse(prod(a), N);
else
  assert(0);
end
  
for i=1:N
  X(i) = 1;

  C_i = convn(X, H, shape);
  C(:, i) = sparse(C_i(:));
  
  X(i) = 0;
end