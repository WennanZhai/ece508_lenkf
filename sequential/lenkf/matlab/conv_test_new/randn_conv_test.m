% randn_conv_test.m

ELEM_TYPE = 'float32';
RANDN_FILENAME = '../tmp/randn';
SETUP_FILENAME = '../tmp/randn_conv_test_setup';
H_FILENAME = '../tmp/h_randn_conv_test_new';
ENSEMBLE_FILENAME = '../tmp/e';

addpath('../');


rand('seed', 1);
randn('seed', 1);

%m = 5;
%n = 2;

L = 4;

%m = 3;
%n = 4;

m = [1 1];
n = [4 4];

%m = [2 2 3];
%n = [2 4 5];

%m = [3 5 2 2 3];
%n = [3 5 3 4 5];

k = n + m - 1;

assert(length(n) == length(m));
rank = length(n);

fid = fopen(SETUP_FILENAME, 'w');
assert(fid ~= -1);

if (rank > 1)
  c = fwrite(fid, [rank [n(end:-1:3) n(1) n(2)] L], 'int32');
else
  c = fwrite(fid, [rank n L], 'int32');
end
assert(c == 1 + rank + 1);
fclose(fid);

h = gen_dist_filter(m, @f);

if (rank == 1)
  h_zp = zeros(1, k);
else
  h_zp = zeros(k);
end

I = cell(1, rank);
for i=1:rank
  I{i} = 1:m(i);
end

h_zp(I{1:end}) = h;

if (rank == 1)
  h_zp = circshift(h_zp, [1, -(m-1)]);
else
  h_zp = circshift(h_zp, -(m-1));
end

h = PI_0_sqrt_sf*a;
h_zp = PI_0_sqrt_sf*a_zp;

export_r_filter_new(H_FILENAME, h_zp, ELEM_TYPE);

H = convmtxn(h, n);
Q_sqrt = H';

w = randn(prod(k), L);

fid = fopen(RANDN_FILENAME, 'w');
assert(fid ~= 1);

for l=1:L
  if (rank > 1)
    w_l = permute(reshape(w(:,l), k), [2, 1, 3:rank]);
  else
    w_l = w(:,l);
  end
  
  c = fwrite(fid, w_l(:), 'double');
  assert(c == prod(size(w_l)));
end
fclose(fid);

%w(:,1)'

e = Q_sqrt * w;


prefix = get_system_prefix(ELEM_TYPE);
cmd = sprintf('%s %s %s %s %s', ...
              prefix, ...
              '../../randn_conv_test_new', ... 
              SETUP_FILENAME, H_FILENAME, ENSEMBLE_FILENAME);
my_system(cmd);

e_c = import_full_r(ENSEMBLE_FILENAME, ELEM_TYPE);

disp(sprintf('error = %e', norm(e(:) - e_c(:))));