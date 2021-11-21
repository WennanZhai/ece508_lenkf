function output_H_cfg_file(cfg);

fid = fopen(cfg.cfg_filename, 'w');

fprintf(fid, 'dir = %s\n', cfg.dir_name);
fprintf(fid, 'H_filename = %s\n', cfg.H_filename);
fprintf(fid, 'res = %d\n', cfg.res);
fprintf(fid, 'na = %d\n', cfg.na);
fprintf(fid, 'np = %d\n', cfg.np);
fprintf(fid, 'd = %d', cfg.d);

fclose(fid);