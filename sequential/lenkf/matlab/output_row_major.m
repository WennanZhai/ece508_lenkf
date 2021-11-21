function output_row_major(fid, A, rank, n, type)

if (nargin == 4)
  type = 'double';
end

  
L = size(A, 2);

for l=1:L
  if (rank > 1)
    a_l = permute(reshape(A(:,l), n), [2, 1, 3:rank]);
  else
    a_l = A(:,l);
  end
  
  c = fwrite(fid, a_l(:), type);
  assert(c == prod(size(a_l)));
end