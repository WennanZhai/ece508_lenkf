function export_diag(filename, A, type)
  
if (nargin == 2)
  type = 'double';
end

fid = fopen(filename, 'w');
assert(fid ~= -1);

c = fwrite(fid, type_to_bytes(type), 'int32');
assert(c == 1);

[m, n] = size(A);

c = fwrite(fid, m, 'int32');
c = c + fwrite(fid, n, 'int32');
assert(c == 2);

c = fwrite(fid, full(diag(A)), type);
assert(c == min(m, n));

fclose(fid);