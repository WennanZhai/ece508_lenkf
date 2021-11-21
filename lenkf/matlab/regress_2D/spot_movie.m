function x = spot_movie(res, T, rad, d, v)
% res - movie is res x res
% T - # of time steps in the movie
% rad - radius of spots (units: pixels)
% d - density of spots (units: spots/pixel)
% v - velocity of spots (units: pixels/time step)


% N is the number of spots
N = ceil(res^2*d);

spot_state = zeros(N, 4);

spot_state(:, 1:2) = rand(N, 2)*(res-1) + 1;

theta = rand(N,1)*(2*pi);

spot_state(:,3) = sin(theta)*v;
spot_state(:,4) = cos(theta)*v;

x = zeros(res^2, T);

%N

for i=1:T
  x_i = zeros(res, res);
  
  for n=1:N
    r = [floor(spot_state(n, 1) - rad), ceil(spot_state(n, 1) + rad)];
    c = [floor(spot_state(n, 2) - rad), ceil(spot_state(n, 2) + rad)];
  
    %spot_state(n,:)
    %r
    %c
    
    for k=r(1):r(2)
      for l=c(1):c(2)
        accum = 1 - sqrt((spot_state(n,1) - k)^2 + ...
                         (spot_state(n,2) - l)^2)/rad;
        
        if (accum > 0)
          row = k;
          col = l;
          
          if (row < 1)
            row = row + res;
          elseif (row > res)
            row = row - res;
          end
          
          if (col < 1)
            col = col + res;
          elseif (col > res)
            col = col - res;
          end
          
          x_i(row, col) = x_i(row, col) + accum;
        end
      end
    end
    
    %x_i
    %pause;
  end
  
  x(:, i) = x_i(:);
  

  spot_state(:, 1) = spot_state(:, 1) + spot_state(:, 3);
  spot_state(:, 2) = spot_state(:, 2) + spot_state(:, 4);
  
  I = find(spot_state(:, 1:2) < 1);
  spot_state(I) = spot_state(I) + res;
  
  I = find(spot_state(:, 1:2) > res);
  spot_state(I) = spot_state(I) - res;
end

x = x - mean(x(:));