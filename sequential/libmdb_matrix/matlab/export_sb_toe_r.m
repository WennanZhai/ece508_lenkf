function export_sb_toe_r(filename, A, type)
% function export_sb_toe_r(filename, A, type)

if (nargin == 2)
  type = 'double';
end

fid = fopen(filename, 'w');
assert(fid ~= -1);

I = find(size(A) > 1);
rank = length(I);

n_phy = nan(1, rank);
X = A;
for i=1:rank
  if (i == rank)
    I = find(X ~= 0);
    n_phy(i) = max(I);
  else
    S = get_sum(X);
    I = find(S ~= 0);
    n_phy(i) = max(I);
    X = get_submatrix(X);
  end
end

s = size(A);
I = find(s > 1);
n = fliplr(s(I));

%rank
%n_phy
%n

c = fwrite(fid, type_to_bytes(type), 'int32');
assert(c == 1);

c = fwrite(fid, rank, 'int32');
assert(c == 1);

c = fwrite(fid, n_phy, 'int32');
assert(c == rank);

c = fwrite(fid, n, 'int32');
assert(c == rank);

s = 'A(';
for i=rank:-1:1
  if (i == 1)
    s = strcat(s, sprintf('1:%d', n_phy(i)));
  else
    s = strcat(s, sprintf('1:%d,', n_phy(i)));
  end
end
s = strcat(s, ');');

%size(A)
%A
%s
B = eval(s);
%B

c = fwrite(fid, full(B(:)), type);
assert(c == length(B(:)));

fclose(fid);



function X = get_submatrix(X)

d = ndims(X);
assert(d >= 2);

s = 'X(';
for j=1:d-1
  s = strcat(s, ':,');
end
s = strcat(s, '1)');

X = eval(s);


function S = get_sum(X)

S = X;
for i=1:ndims(X) - 1
  S = sum(S);
end

S = squeeze(S);