function export_vector(filename, v, type)
  
if (nargin == 2)
  type = 'double';
end

fid = fopen(filename, 'w');
assert(fid ~= -1);

N = length(v);

c = fwrite(fid, type_to_bytes(type), 'int32');
assert(c == 1);

c = fwrite(fid, N, 'int32');
assert(c == 1);

c = fwrite(fid, v, type);
assert(c == N);

fclose(fid);