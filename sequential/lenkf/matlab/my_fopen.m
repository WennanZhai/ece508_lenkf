function fid = my_fopen(fname, mode)
% ction fid = my_fopen(fname, mode)

fid = fopen(fname, mode);

if (fid == -1)
  error(['Error opening file: ', fname]);
end