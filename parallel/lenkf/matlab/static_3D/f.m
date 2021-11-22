function y = f(x, l)

I = find(x >= l);
J = find(x < l);

y(I) = 0;
y(J) = sqrt(l - x(J) + 1);

%y(J) = x(J).^2;
