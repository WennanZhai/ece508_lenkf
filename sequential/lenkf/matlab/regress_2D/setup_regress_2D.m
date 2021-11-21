% setup_regress_2D.m

addpath('../');

cfg = struct();
cfg.dir_name = '../tmp';
cfg.H_filename = 'H_regress_2D_unscaled';
cfg.res = res;
cfg.na = na;
cfg.np = np;
cfg.d = d;

generate_H_matrix(cfg);

H = import_rcs(sprintf('%s/%s', cfg.dir_name, cfg.H_filename));
%s = svd(H);
%H = H/s(1);
H = H / svds(H, 1);
H = H * beta;

M = np;
N = res^2;

x = spot_movie(res, T, rad, d, v);
x = x * sqrt(alpha);

y_true = zeros(M, T);
y = zeros(M, T);

H_row = 1;

for i=1:T
  H_index = np*(H_row - 1) + 1;
  H_I = H_index:H_index+np-1;
  y_true(:, i) = H(H_I, :) * x(:, i);
  H_row = H_row + 1;

  if (H_row > size(H,1)/M)
    H_row = 1;
  end
end

% COMPLETELY IGNORED LATER IN THE CODE IF POISSON_NOISE IS SET
var_y = sum(y_true(:).^2)/(prod(size(y_true))-1);
var_v = var_y/SNR;
sig_v = sqrt(var_v);

if (POISSON_NOISE)
  for i=1:T
    y(:,i) = poissrnd(y_true(:,i));
  end
else
  v = sig_v*randn(size(y_true));
  y = y_true + v;
end
