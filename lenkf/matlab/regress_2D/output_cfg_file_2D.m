function output_cfg_file_2D(cfg)

fid = fopen(cfg.cfg_filename, 'w');
assert(fid ~= -1);

fprintf(fid, 'dir = %s\n', cfg.dir_name);

fprintf(fid, 'x_hat_filename = %s\n', cfg.x_hat_filename);
if (cfg.save_trace)
  fprintf(fid, 'save_trace = true\n');
  fprintf(fid, 'trace_filename = %s\n', cfg.trace_filename);
else
  fprintf(fid, 'save_trace = false\n');
end

fprintf(fid, 'rank = %d\n', cfg.rank);
fprintf(fid, 'n = {');
for i = 1:cfg.rank-1
  fprintf(fid, '%d, ', cfg.n(i));
end
fprintf(fid, '%d}\n', cfg.n(cfg.rank));

fprintf(fid, 'N = %d\n', cfg.N);
fprintf(fid, 'M_block = %d\n', cfg.M_block);
fprintf(fid, 'T = %d\n', cfg.T);
fprintf(fid, 'L = %d\n', cfg.L);
if (cfg.regularize)
  fprintf(fid, 'regularize = true\n');
  fprintf(fid, 'lambda = %f\n', cfg.lambda);
else
  fprintf(fid, 'regularize = false\n');
end
if (cfg.quiet_mode)
  fprintf(fid, 'quiet_mode = true\n');
else
  fprintf(fid, 'quiet_mode = false\n');
end
if (cfg.randn_debug)
  fprintf(fid, 'randn_debug = true\n');
else
  fprintf(fid, 'randn_debug = false\n');
end
if (cfg.randn_conv)
  fprintf(fid, 'randn_conv = true\n');
else
  fprintf(fid, 'randn_conv = false\n');
end
if (cfg.lenkf_debug)
  fprintf(fid, 'lenkf_debug = true\n');
else
  fprintf(fid, 'lenkf_debug = false\n');
end

if (cfg.update_epsilon == 0)
  fprintf(fid, 'update_epsilon = 0\n', cfg.update_epsilon);
else
  fprintf(fid, 'update_epsilon = %f\n', cfg.update_epsilon);
end

fprintf(fid, 'F_equal_I = true\n');

fprintf(fid, 'x0_filename = %s\n', cfg.x0_filename);
fprintf(fid, 'PI_sqrt_filename = %s\n', cfg.PI_sqrt_filename);
fprintf(fid, 'D_filename = %s\n', cfg.D_filename);
fprintf(fid, 'Q_sqrt_filename = %s\n', cfg.Q_sqrt_filename);
fprintf(fid, 'C_filename = %s\n', cfg.C_filename);

fprintf(fid, 'y_list_filename = %s\n', cfg.y_list_filename);
fprintf(fid, 'H_list_filename = %s\n', cfg.H_list_filename);
fprintf(fid, 'R_sqrt_list_filename = %s', cfg.R_sqrt_list_filename);

fclose(fid);
