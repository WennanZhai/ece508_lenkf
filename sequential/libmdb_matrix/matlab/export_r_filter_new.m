function export_r_filter_new(filename, A, type)

if (nargin == 2)
  type = 'double';
end


fid = fopen(filename, 'w');
assert(fid ~= -1);

S = size(A);
I = find(S > 1);
rank = length(I);

c = fwrite(fid, type_to_bytes(type), 'int32');
assert(c == 1);

count = fwrite(fid, rank, 'int32');
assert(count == 1);

n_log = size(A);
I = find(n_log > 1);
n_log = n_log(I);

if (rank >= 2)
  n_log = fliplr(n_log);
  n_log(end-1:end) = n_log(end:-1:end-1);
end

count = fwrite(fid, n_log, 'int32');
assert(count == rank);

if (rank >= 2)
  I = 1:rank;
  I(1) = 2;
  I(2) = 1;
  A = permute(A, I);
end

count = fwrite(fid, full(A(:)), type);
assert(count == prod(n_log));