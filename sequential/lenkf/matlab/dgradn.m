function [varargout] = dgradn(n)

rank = length(n);

D = cell(1, rank);

for i=1:rank
  s = ones(1, rank);
  s(i) = 2;
  
  H = zeros(s);
  H(1) = 1;
  H(2) = -1;
  
  D{i} = convmtxn(H, n, 'valid');
end

if (rank > 1)
  D_1 = D{1};
  D{1} = D{2};
  D{2} = D_1;
end

varargout = D;