% setup_static_2D.m

addpath('../');

cfg = struct();
cfg.cfg_filename = '../tmp/H_static_2D.cfg';
cfg.dir_name = '../tmp';
cfg.H_filename = 'H_static_2D_unscaled';
cfg.res = res;
cfg.na = na;
cfg.np = np;
cfg.d = d;

output_H_cfg_file(cfg);

cmd = sprintf('../../../sysmatrix/sysmatrix %s', cfg.cfg_filename);

[s, w] = system(cmd);
assert(s == 0);

H = import_rcs(sprintf('%s/%s', cfg.dir_name, cfg.H_filename), 'float32');
%s = svd(H);
%H = H/s(1);
H = H / svds(H, 1);


x = phantom(res);
x = x(:);

y_true = zeros(M, T);
y = zeros(M, T);

H_row = 1;

for i=1:T
  H_index = M*(H_row - 1) + 1;
  H_I = H_index:H_index+M-1;
  y_true(:, i) = H(H_I, :) * x(:, i);
  H_row = H_row + 1;
  
  if (H_row > size(H,1)/M)
    H_row = 1;
  end
end

var_v = sum(y_true(:).^2) * 10^(-NSNR/10) / prod(size(y));
sig_v = sqrt(var_v);

v = sig_v*randn(size(y_true));
y = y_true + v;