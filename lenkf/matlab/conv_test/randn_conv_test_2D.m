% randn_conv_test_2D.m

ELEM_TYPE = 'float32';

addpath('../');


res = 8;
K_phy = 2;
K_log = K_phy * 2;

[Q, B, A] = gen_Q(res, K_log);

P = 4*1024;



N = res;
K = N + K_log - 1;
if (mod(K,2) == 1)
  K = K + 1;
  A_zp = [A, zeros(K_log, 1);
          zeros(1, K_log + 1)];
else
  A_zp = A;
end

C_sqrt = convmtx2(A_zp, N, N);
Q_sqrt = C_sqrt';
Q = Q_sqrt*Q_sqrt';

n = randn(K^2,P);
u_conv = nan(N^2,P);


%%% Convolution technique

fid = my_fopen('../tmp/randn', 'w');
for p=1:P
  n_p = n(:, p);
  c = fwrite(fid, n_p(:), 'double');
  assert(c == length(n_p(:)));
end
fclose(fid);

K_phy = K_log/2;
A_s = A(K_phy+1:end, K_phy+1:end);
A_s_zp = [A_s, zeros(K_phy, K-K_log), fliplr(A_s)];
A_s_zp = [A_s_zp; zeros(K-K_log, K); flipud(A_s_zp)];

A_F = fft2(A_s_zp);

offset = K_phy - 1;
for p=1:P
  n_p = reshape(n(:, p), K, K);
  N_p = fft2(n_p, K, K);
  
  y = ifft2(N_p .* A_F);
  
  y = y((1:N) + offset, (1:N) + offset);
  u_conv(:, p) = y(:);
end

Cu_conv = cov(u_conv');


%%% Convolution matrix technique

u_convmtx = Q_sqrt*n;

Cu_convmtx = cov(u_convmtx');

disp(sprintf('max(abs(u_conv(:) - u_convmtx(:))) = %f', ...
             max(abs(u_conv(:) - u_convmtx(:)))));


%%% C implementation of matrix convolution and convolution technique

export_rcs('../tmp/Q_sqrt_test_2D', Q_sqrt, ELEM_TYPE);
export_full_r('../tmp/h_psf_test', A_s, ELEM_TYPE);


if (strcmp(ELEM_TYPE, 'float32'))
  cmd = 'LD_PRELOAD=/usr/lib/libfftw3f.so ../../randn_conv_test_2D';
elseif (strcmp(ELEM_TYPE, 'double'))
  cmd = 'LD_PRELOAD=/usr/lib/libfftw3.so ../../randn_conv_test_2D';
else
  assert(0);
end

my_system(cmd);

e_convmtx = import_full_r('../tmp/e_convmtx', ELEM_TYPE);
e_conv = import_full_r('../tmp/e_conv', ELEM_TYPE);
e_unit = import_full_r('../tmp/e_unit', ELEM_TYPE);

Cu_convmtx_c = cov(e_convmtx');
Cu_conv_c = cov(e_conv');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

figure(1);
subplot(1,3,1);
imagesc(Q);
axis('image');
title('Q');
subplot(1,3,2);
imagesc(Cu_conv);
title('conv');
axis('image');
subplot(1,3,3);
imagesc(Cu_conv_c);
title('conv c');
axis('image');

figure(2);
subplot(1,2,1);
imagesc(abs(Q - Cu_conv));
axis('image');
title('abs(Q - Cu\_conv)');
colorbar;
subplot(1,2,2);
imagesc(abs(Cu_conv - Cu_conv_c));
axis('image');
title('abs(Cu\_conv - Cu\_conv\_c)');
colorbar;

figure(3);
subplot(1,3,1);
imagesc(Q);
axis('image');
title('Q');
subplot(1,3,2);
imagesc(Cu_convmtx);
title('convmtx');
axis('image');
subplot(1,3,3);
imagesc(Cu_convmtx_c);
title('convmtx c');
axis('image');

figure(4);
subplot(1,2,1);
imagesc(abs(Q - Cu_convmtx));
axis('image');
title('abs(Q - Cu\_convmtx)');
colorbar;
subplot(1,2,2);
imagesc(abs(Cu_convmtx - Cu_convmtx_c));
axis('image');
title('abs(Cu\_convmtx - Cu\_convmtx\_c)');
colorbar;

figure(5);
subplot(1,3,1);
qqplot(e_unit(:));
axis('square');
title('N(0,1)');
subplot(1,3,2);
qqplot(e_conv(:));
axis('square');
title('N(0,Q) via conv');
subplot(1,3,3);
qqplot(e_convmtx(:));
axis('square');
title('N(0,Q) via conv mtx');