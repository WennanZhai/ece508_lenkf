function A = import_rcs(filename)
  
fid = fopen(filename, 'r');
assert(fid ~= -1);

bytes = fread(fid, 1, 'int32');

[scratch] = fread(fid, 3, 'int32');
m = scratch(1);
n = scratch(2);
N = scratch(3);

v = zeros(N,1);
j = zeros(N,1);
r = zeros(m+1,1);

[v, count] = fread(fid, N, bytes_to_type(bytes));
assert(count == N);

[j, count] = fread(fid, N, 'int32');
assert(count == N);

[r, count] = fread(fid, m+1, 'int32');
assert(count == m+1);

fclose(fid);

i = zeros(size(j));
count = 1;

for c = 1:m
  num_on_row = r(c+1) - r(c);
  
  if (num_on_row ~= 0)
    i(count:count+num_on_row-1) = c*ones(num_on_row, 1);
    count = count+num_on_row;
  end
end

%max(i)
%max(j)

%i
%j
%v
%r

A = sparse(i, j+1, v, m, n);
