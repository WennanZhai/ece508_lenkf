% randn_conv_test.m

ELEM_TYPE = 'float32';

addpath('../');

K_phy = 2;
K_log = K_phy*2;
N = 32;
P = 2^10;


h = triang(K_log);

q = conv(h, h);

Q = toeplitz([q(K_log:end); zeros(N - K_log, 1)]);


%%% Cholesky technique

Q_sqrt = chol(Q)';

n = randn(N, P);
u = Q_sqrt*n;

fid = my_fopen('../tmp/randn', 'w');
c = fwrite(fid, n, ELEM_TYPE);
assert(c == length(n(:)));

Cu_chol = cov(u');


%%% Convolution technique

K = N+K_log-1;

if (mod(K,2) == 1)
  K = K + 1;
  h_zp = [h; 0];
else
  h_zp = h;
end

h_pad = [h(K_log/2+1:end); zeros(K-K_log,1); h(1:K_log/2)];
H = fft(h_pad);

n = randn(K, P);
u_conv = zeros(N, P);

c = fwrite(fid, n, ELEM_TYPE);
assert(c == length(n(:)));
fclose(fid);

offset = K_phy - 1;
for p=1:P
  y = ifft(H.*fft(n(:,p)));
  u_conv(:, p) = y((1:N) + offset);
end

Cu_conv = cov(u_conv');


%%% Convolution matrix technique

Q_sqrt_convmtx = convmtx(h_zp, N)';
u_convmtx = Q_sqrt_convmtx * n;

disp(sprintf('max(abs(u_conv(:) - u_convmtx(:))) = %f', ...
             max(abs(u_conv(:) - u_convmtx(:)))));


%%% C implementation of Cholesky and convolution technique

export_rcs('../tmp/Q_sqrt_test', Q_sqrt, ELEM_TYPE);

export_vector('../tmp/h_test', h(K_log/2+1:end), ELEM_TYPE);


if (strcmp(ELEM_TYPE, 'float32'))
  cmd = 'LD_PRELOAD=/usr/lib/libfftw3f.so ../../randn_conv_test';
elseif (strcmp(ELEM_TYPE, 'double'))
  cmd = 'LD_PRELOAD=/usr/lib/libfftw3.so ../../randn_conv_test';
else
  assert(0);
end

my_system(cmd);

e_chol = import_full_r('../tmp/e_chol', ELEM_TYPE);
e_conv = import_full_r('../tmp/e_conv', ELEM_TYPE);


Cu_chol_c = cov(e_chol');
Cu_conv_c = cov(e_conv');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

figure(1);
subplot(1,3,1);
imagesc(Q);
axis('image');
title('Q');
subplot(1,3,2);
imagesc(Cu_chol);
axis('image');
title('chol');
subplot(1,3,3);
imagesc(Cu_conv);
axis('image');
title('conv');


figure(2);
subplot(1,2,1);
imagesc(abs(Q - Cu_chol));
axis('image');
title('abs(Q - chol)');
colorbar;
subplot(1,2,2);
imagesc(abs(Q - Cu_conv));
axis('image');
title('abs(Q - conv)');
colorbar;

figure(3);
subplot(1,3,1);
imagesc(Q);
axis('image');
title('Q');
subplot(1,3,2);
imagesc(Cu_chol_c);
axis('image');
title('chol c');
subplot(1,3,3);
imagesc(Cu_conv_c);
axis('image');
title('conv c');

figure(4);
subplot(1,2,1);
imagesc(abs(Cu_chol - Cu_chol_c));
axis('image');
title('abs(Cu\_chol - Cu\_chol\_c)');
colorbar;
subplot(1,2,2);
imagesc(abs(Cu_conv - Cu_conv_c));
axis('image');
title('abs(Cu\_conv - Cu\_conv\_c)');
colorbar;