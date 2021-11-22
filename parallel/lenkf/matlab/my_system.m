function [s,w] = my_system(command, path)
% [s,w] = my_system(command, [path])

if (nargin == 2)
  pd = pwd;
  
  cd(path);
  [s,w] = system(command);
  cd(pd);
else
  path = pwd;
  [s,w] = system(command);
end

if (s ~= 0)
  error(['Error executing command: ', command, ' (dir: ', path, ')']);
end