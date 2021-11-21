function fprintf_full(fid, A)

[M,N] = size(A);

for i=1:M
  l = '';
  for j=1:N
    l = sprintf('%s%s', l, printf_elem_s(A(i,j)));;
  end
  fprintf(fid, '%s\n', l);
end
