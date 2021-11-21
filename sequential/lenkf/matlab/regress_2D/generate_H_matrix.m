function generate_H_matrix(cfg);
    axis_lim = cfg.res * cfg.d / 2;

    H_filename = fullfile(cfg.dir_name, cfg.H_filename);

    cmd = sprintf('/usr/local/bin/python3 -m pyinverse.radon %s -n %d -a %d -p %d --xlim %f %f --ylim %f %f --tlim %f %f', ...
                  H_filename, cfg.res, cfg.na, cfg.np, ...
                  -axis_lim, axis_lim, ...
                  -axis_lim, axis_lim, ...
                  -sqrt(2)*axis_lim, sqrt(2)*axis_lim);

    disp(cmd);
    [s, w] = system(cmd);
    assert(s == 0);

    H_npz_filename = strcat(H_filename, '.npz');

    cmd = sprintf('./convert_npz.py %s %s', H_filename, H_npz_filename);
    [s, w] = system(cmd);
    assert(s == 0);
end
