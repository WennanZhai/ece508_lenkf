% setup_static_3D.m

addpath('../');

cfg = struct();

cfg.cfg_filename = '../tmp/static_3D.cfg';

cfg.dir = '../tmp/';
cfg.suffix = 'static_3D';

cfg.n_theta = n_theta;
cfg.n_phi = n_phi;
cfg.n_u = n_u;
cfg.n_v = n_v;

cfg.n_x = res;
cfg.n_y = res;
cfg.n_z = res;

output_cfg_file_radon3D(cfg);

cmd = sprintf('../../../radon3D/radon3D %s', cfg.cfg_filename);

[s,r] = system(cmd);

assert(s == 0);


v_filename = sprintf('%sv_%s', cfg.dir, cfg.suffix);
j_filename = sprintf('%sj_%s', cfg.dir, cfg.suffix);
r_filename = sprintf('%sr_%s', cfg.dir, cfg.suffix);

H = import_rcs3(v_filename, j_filename, r_filename, 'float32');
H = H / svds(H, 1);

x = phantom3d(res);
x = x(:);

y_true = zeros(M, T);
y = zeros(M, T);

H_row = 1;

for i=1:T
  H_I = H_row:H_row+M-1;
  y_true(:, i) = H(H_I, :) * x;
  H_row = H_row + M;
  
  if (H_row > size(H,1))
    H_row = 1;
  end
end

var_v = sum(y_true(:).^2) * 10^(-NSNR/10) / prod(size(y));
sig_v = sqrt(var_v);

v = sig_v*randn(size(y_true));
y = y_true + v;
