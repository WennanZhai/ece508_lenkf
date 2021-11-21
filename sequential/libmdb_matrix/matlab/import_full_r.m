function A = import_full_r(filename)
  
fid = fopen(filename, 'r');
assert(fid ~= -1);

bytes = fread(fid, 1, 'int32');

m = fread(fid, 1, 'int32');
n = fread(fid, 1, 'int32');

A = zeros(m, n);

for i=1:m
  for j=1:n
    A(i,j) = fread(fid, 1, bytes_to_type(bytes));
  end
end

fclose(fid);