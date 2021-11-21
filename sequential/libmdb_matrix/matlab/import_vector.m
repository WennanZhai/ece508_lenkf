function v = import_vector(filename)

fid = fopen(filename, 'r');
assert(fid ~= -1);

bytes = fread(fid, 1, 'int32');

[n, count] = fread(fid, 1, 'int32');
assert(count == 1);

[v, count] = fread(fid, bytes_to_type(bytes));
assert(count == n);

fclose(fid);