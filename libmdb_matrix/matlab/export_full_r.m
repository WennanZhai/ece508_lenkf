function export_full_r(filename, A, type)

if (nargin == 2)
  type = 'double';
end

fid = fopen(filename, 'w');
assert(fid ~= -1);

[m, n] = size(A);

A = A';
v = A(:);

c = fwrite(fid, type_to_bytes(type), 'int32');
assert(c == 1);

c = fwrite(fid, type_to_bytes(type), 'int32');
assert(c == 1);

c = fwrite(fid, m, 'int32');
assert(c == 1);

c = fwrite(fid, n, 'int32');
assert(c == 1);

c = fwrite(fid, v, type);
assert(c == m*n);

fclose(fid);