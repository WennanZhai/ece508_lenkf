function H = gen_dist_filter(n, shape_fun)

rank = length(n);
assert(rank >= 1);

X = cell(1,rank);
Y = cell(1,rank);

for i=1:rank
  X{i} = linspace(-(n(i)-1)/2, (n(i)-1)/2, n(i));
end

if (rank == 1)
  Y{1} = X{1};
else
  [Y{1:end}] = ndgrid(X{1:end});
end

if (rank == 1)
  D = zeros(1, n);
else
  D = zeros(n);
end

for i=1:rank
  D = D + Y{i}.^2;
end

D = sqrt(D);

H = reshape(shape_fun(D(:)), size(D));