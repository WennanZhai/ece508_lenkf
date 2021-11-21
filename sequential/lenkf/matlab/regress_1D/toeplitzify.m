function B = toeplitzify(A)

[m,n] = size(A);

assert(m == n)

r = zeros(m,1);

for i=0:m-1
    if (i > 0)
        r(i+1) = mean(diag(A, i) + diag(A,-i))/2;
    else
        r(1) = mean(diag(A, 0));
    end
end

B = toeplitz(r);