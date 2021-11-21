function C = gc(z, c)
% Implementation of equation 4.10 from G&C


l_z = length(z);
C = zeros(l_z, 1);

az = abs(z);
  
I = find(az <= c);
C(I) = -1/4*(az(I)/c).^5 + 1/2*(az(I)/c).^4 + 5/8*(az(I)/c).^3 - ...
       5/3*(az(I)/c).^2 + 1;

I = find(az >= c & az <= 2*c);
C(I) = 1/12*(az(I)/c).^5 -1/2*(az(I)/c).^4 + 5/8*(az(I)/c).^3 + ...
       5/3*(az(I)/c).^2 - 5*(az(I)/c) + 4 - 2/3*c./az(I);