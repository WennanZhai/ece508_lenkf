function export_rcs(filename, A, type)
  
if (nargin == 2)
  type = 'double';
end

fid = fopen(filename, 'w');
assert(fid ~= -1);

[m, n] = size(A);
N = nnz(A);

c = fwrite(fid, type_to_bytes(type), 'int32');
assert(c == 1);

c = fwrite(fid, m, 'int32');
c = c + fwrite(fid, n, 'int32');
c = c + fwrite(fid, N, 'int32');
assert(c == 3);

A_T = A';
I2 = find(A_T);
[I,J,scratch] = find(A);

c = fwrite(fid, full(A_T(I2)), type);
assert(c == length(I2));

j = mod(I2-1, n);
c = fwrite(fid, j, 'int32');
assert(c == length(I2));

r = zeros(m, 1);

for i=1:length(I)
  r(I(i)) = r(I(i)) + 1;
end

r = [0; cumsum(r)];
c = fwrite(fid, r, 'int32');
assert(c == m+1);

fclose(fid);