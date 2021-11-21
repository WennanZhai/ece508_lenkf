function output_cfg_file_radon3D(cfg)

fid = fopen(cfg.cfg_filename, 'w');

fprintf(fid, 'dir = "%s"\n', cfg.dir);
fprintf(fid, 'suffix = "%s"\n', cfg.suffix);

fprintf(fid, 'n_theta = %d\n', cfg.n_theta);
fprintf(fid, 'n_phi = %d\n', cfg.n_phi);
fprintf(fid, 'n_u = %d\n', cfg.n_u);
fprintf(fid, 'n_v = %d\n', cfg.n_v);

fprintf(fid, 'n_x = %d\n', cfg.n_x);
fprintf(fid, 'n_y = %d\n', cfg.n_y);
fprintf(fid, 'n_z = %d', cfg.n_z);

fclose(fid);