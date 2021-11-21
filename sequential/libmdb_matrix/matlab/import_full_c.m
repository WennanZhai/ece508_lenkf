function A = import_full_c(filename)

fid = fopen(filename, 'r');
assert(fid ~= -1);

bytes = fread(fid, 1, 'int32');

m = fread(fid, 1, 'int32');
n = fread(fid, 1, 'int32');

A = fread(fid, m * n, bytes_to_type(bytes));
A = reshape(A, m, n);

fclose(fid);