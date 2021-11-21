% static_2D.m

lsq        = 1;
kf         = 1;
lskf       = 0;
lvkf       = 1;
lenkf      = 1;
c_lenkf    = 1;

ELEM_TYPE = 'float32';

PLOT_TRACE = 1;

LENKF_DEBUG = 0;
LENKF_DEBUG_FILENAME= '../tmp/static_2D.debug';

RANDN_DEBUG = 1;
RANDN_CONV  = 1;
RANDN_FILENAME = '../tmp/randn';

res = 16;
regularize = 1;
lambda = 1e4;

localize = 1;
%taper = res - 1;
taper = 0;

update_epsilon = 0;

na = res;
np = ceil(sqrt(2)*res);
d = 1;
NSNR = 50;

T = 1;
M = np*na;
M_block = res^2;
%M_block = 1;
N = res^2;
%L = 2^12;
L = 256;

rank = 2;
n = [res, res];
n_output = [n(end:-1:3) n(1) n(2)];

corr_l = 3;
Q_sqrt_sf = 1;
PI_0_sqrt_sf = 1e-1;


%if (RANDN_DEBUG || LENKF_DEBUG)
  rand('seed', 1);
  randn('seed', 1);
  %end


% Creates y and H
setup_static_2D;

F = speye(N,N);

R = var_v*speye(M,M);
R_sqrt = sig_v*speye(M,M);

x0 = zeros(N,1);

if (regularize)
  [Dx, Dy] = dgrad([res, res]);
  D = [Dx; Dy];
  D = D / sqrt(svds(Dx,1)^2 + svds(Dy,1)^2);
else
  D = zeros(0);
end
[D_M, D_N] = size(D);

% Creates a (sqrt distance filter), a_zp (zero-padded a), and A (convolution
% matrx form of a)
do_gen_dist_filter;

Q_sqrt_convmtx = Q_sqrt_sf * A';
Q = Q_sqrt_convmtx*Q_sqrt_convmtx';
Q_sqrt = chol(Q)';
Q_sqrt_zp = Q_sqrt_sf * a_zp;

PI_0_sqrt_convmtx = PI_0_sqrt_sf * A';
PI_0 = PI_0_sqrt_convmtx*PI_0_sqrt_convmtx';
PI_0_sqrt = chol(PI_0)';
PI_0_sqrt_zp = PI_0_sqrt_sf * a_zp; 

J = size(Q_sqrt_convmtx, 2);


N_to_plot = sum([lsq kf lskf lvkf lenkf c_lenkf]);

to_plot = zeros(N,T,N_to_plot);
to_plot_t = {};
to_plot_index = 1;

traces = zeros(N_to_plot, T);



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  Least squares

if (lsq)
  R_sqrt_inv = inv(R_sqrt);
  Z = cell(1,T);
  for i=1:T
    Z{i} = R_sqrt_inv;
  end
  R_sqrt_inv_blkdiag = blkdiag(Z{:});

  A = [R_sqrt_inv_blkdiag*H;
       inv(PI_0_sqrt)];
    
  b = [R_sqrt_inv_blkdiag*y(:);
       inv(PI_0_sqrt)*x0];

  if (regularize)
    A = [A; sqrt(lambda)*D];
    b = [b; zeros(D_M,1)];
  end
  
  x_lsq = A\b;

  if (0)
  if (PLOT_TRACE)
    for i=1:T
      traces(to_plot_index, i) = nan;
    end
  end
  end
  
  to_plot(:,to_plot_index) = x_lsq;
  to_plot_index = to_plot_index + 1;
  to_plot_t(end+1) = {'ls'};
  
  d = x_lsq - x;
  e_lsq = norm(d(:))/norm(x(:));
  
  disp(sprintf('ls:\t\t%f', e_lsq));
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  Kalman filter

if (kf)
  row_H = 1;
  
  x_kf_f = zeros(N,T);   % The filtered estimate  x_{i|i}
  x_kf_p = zeros(N,T+1); % The predicted estimate x_{i|i-1}
  x_kf_p(:,1) = x0;
  
  P_p = PI_0;            % Initial state covariance
  P_f = zeros(N,N);      % The filtered error covariance  P_{i|i}
  
  for i=1:T
    H_I = row_H:row_H+M-1;
    
    % Measurement update
    %K = P_p*H(H_I,:)'*inv(H(H_I,:)*P_p*H(H_I,:)' + R);
    %x_kf_f(:,i) = x_kf_p(:,i) + K*(y(:,i) - H(H_I,:)*x_kf_p(:,i));
    %P_f = P_p - K*H(H_I,:)*P_p;
    
    B = H(H_I,:)*P_p*H(H_I,:)' + R;
    x_kf_f(:,i) = x_kf_p(:,i) + ...
        P_p*H(H_I,:)'*(B\(y(:,i) - H(H_I,:)*x_kf_p(:,i)));
    P_f = P_p - P_p*H(H_I,:)'*(B\(H(H_I,:)*P_p));
    
    row_H = row_H + M;
    if (row_H > size(H,1))
      row_H = 1;
    end

    if (regularize)
      %K = P_f*D'*inv(D*P_f*D' + 1/lambda*eye(D_M));
      %x_kf_f(:,i) = x_kf_f(:,i) - K*D*x_kf_f(:,i);
      %P_f = P_f - K*D*P_f;
      
      B = D*P_f*D' + 1/lambda*eye(D_M);
      x_kf_f(:,i) = x_kf_f(:,i) - ...
          P_f*D'*(B\(D*x_kf_f(:,i)));
      P_f = P_f -  P_f*D'*(B\(D*P_f));
    end

    if (0)
    if (PLOT_TRACE)
      traces(to_plot_index, i) = trace(P_f);
    end
    end
    
    % Time update
    x_kf_p(:,i+1) = F*x_kf_f(:,i);
    P_p = F*P_f*F' + Q;
  end
  
  to_plot(:,:,to_plot_index) = x_kf_f;
  to_plot_index = to_plot_index + 1;
  to_plot_t(end+1) = {'kf'};
  
  d = x_kf_f - x;
  e_kf = norm(d(:))/norm(x(:));
  
  disp(sprintf('kf:\t\t%f', e_kf));
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  Generate localization mask

if (localize)
  C = gen_gc_mask_2D(res, taper);
else
  C = ones(N, N);
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  Localized serial Kalman filter

if (lskf)
  row_H = 1;
  
  x_lskf_f = zeros(N,T);   % The filtered estimate  x_{i|i}
  x_lskf_p = zeros(N,T+1); % The predicted estimate x_{i|i-1}
  x_lskf_p(:,1) = x0;
  
  P_p = zeros(N,N);      % The predicted error covariance P_{i|i-1}
  P_f = zeros(N,N);      % The filtered error covariance  P_{i|i}
  
  P_p = PI_0;

  for i=1:T
    % Measurement update
    x_lskf_f(:,i) = x_lskf_p(:,i);
    P_f = P_p;

    for m=1:M
      k = (C.*P_f)*H(row_H,:)'/...
          (H(row_H,:)*(C.*P_f)*H(row_H,:)' + R(m,m));
      x_lskf_f(:,i) = x_lskf_f(:,i) + k*(y(m,i) - H(row_H,:)*x_lskf_f(:,i));
      P_f = P_f - k*H(row_H,:)*P_f - P_f*H(row_H,:)'*k' + ...
            k*(H(row_H,:)*P_f*H(row_H,:)' + R(m,m))*k';
      
      row_H = row_H + 1;
      if (row_H > size(H,1))
        row_H = 1;
      end
    end
    
    if (regularize)
      for m=1:D_M
        k = (C.*P_f)*D(m,:)'/...
            (D(m,:)*(C.*P_f)*D(m,:)' + 1/lambda);
        x_lskf_f(:,i) = x_lskf_f(:,i) - k*D(m,:)*x_lskf_f(:,i);
        P_f = P_f - k*D(m,:)*P_f - P_f*D(m,:)'*k' + ...
              k*(D(m,:)*P_f*D(m,:)' + 1/lambda)*k';
      end
    end
    
    if (0)
    if (PLOT_TRACE)
      traces(to_plot_index, i) = trace(P_f);
    end
    end
    
    % Time update
    x_lskf_p(:,i+1) = F*x_lskf_f(:,i);
    P_p = F*P_f*F' + Q;
  end
  
  
  to_plot(:,:,to_plot_index) = x_lskf_f;
  to_plot_index = to_plot_index + 1;
  to_plot_t(end+1) = {'lskf'};
  
  d = x_lskf_f - x;
  e_lskf = norm(d(:))/norm(x(:));

  disp(sprintf('lskf:\t\t%f', e_lskf));
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  Localized vector Kalman filter

if (lvkf)
  row_H = 1;
  
  x_lvkf_f = zeros(N,T);   % The filtered estimate  x_{i|i}
  x_lvkf_p = zeros(N,T+1); % The predicted estimate x_{i|i-1}
  x_lvkf_p(:,1) = x0;
  
  P_p = zeros(N,N);      % The predicted error covariance P_{i|i-1}
  P_f = zeros(N,N);      % The filtered error covariance  P_{i|i}
  
  P_p = PI_0;

  for i=1:T
    % Measurement update
    x_lvkf_f(:,i) = x_lvkf_p(:,i);
    P_f = P_p;

    for m=1:M_block:M
      if (m+M_block > M) 
        n_rows = M - m + 1;
      else
        n_rows = M_block;
      end
      
      B_I = m:m+n_rows-1;
      H_I = row_H:row_H+n_rows-1;

      K = (C.*P_f)*H(H_I,:)'...
          *inv(H(H_I,:)*(C.*P_f)*H(H_I,:)' + R(B_I, B_I));
      x_lvkf_f(:,i) = x_lvkf_f(:,i) + K*(y(B_I,i) - H(H_I,:)*x_lvkf_f(:,i));
      P_f = P_f - K*H(H_I,:)*P_f - P_f*H(H_I,:)'*K' + ...
            K*(H(H_I,:)*P_f*H(H_I,:)' + R(B_I,B_I))*K';

      row_H = row_H + n_rows;
      if (row_H > size(H,1))
        assert(row_H == size(H,1) + 1);
        row_H = 1;
      end
    end
    
    if (regularize)
      row_D = 1;
      
      for m=1:M_block:size(D,1)
        if (m+M_block > size(D,1))
          n_rows = size(D,1) - m + 1;
        else
          n_rows = M_block;
        end

        D_I = row_D:row_D+n_rows-1;
       
        K = (C.*P_f)*D(D_I,:)'...
            *inv(D(D_I,:)*(C.*P_f)*D(D_I,:)' + 1/lambda*eye(n_rows, n_rows));
        x_lvkf_f(:,i) = x_lvkf_f(:,i) - K*D(D_I,:)*x_lvkf_f(:,i);
        P_f = P_f - K*D(D_I,:)*P_f - P_f*D(D_I,:)'*K' + ...
              K*(D(D_I,:)*P_f*D(D_I,:)' + 1/lambda*eye(n_rows, n_rows))*K';
        
        row_D = row_D + n_rows;
      end
    end

    if (0)
    if (PLOT_TRACE)
      traces(to_plot_index, i) = trace(P_f);
    end
    end
    
    % Time update
    x_lvkf_p(:,i+1) = F*x_lvkf_f(:,i);
    P_p = F*P_f*F' + Q;
  end
  
  
  to_plot(:,:,to_plot_index) = x_lvkf_f;
  to_plot_index = to_plot_index + 1;
  to_plot_t(end+1) = {'lvkf'};
  
  d = x_lvkf_f - x;
  e_lvkf = norm(d(:))/norm(x(:));

  disp(sprintf('lvkf:\t\t%f', e_lvkf));
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  Localized ensemble Kalman filter

if (lenkf)
  if (RANDN_DEBUG)
    randn_fid = my_fopen(RANDN_FILENAME, 'w');
  end
  
  if (LENKF_DEBUG)
    debug_fid = fopen(LENKF_DEBUG_FILENAME, 'w');
  end
  
  x_lenkf_f = zeros(N,T);   % The filtered estimate  x_{i|i}

  if (RANDN_CONV)
    V = randn(J,L);
  else
    V = randn(N,L);
  end

  if (RANDN_DEBUG)
    if (RANDN_CONV)
      output_row_major(randn_fid, V, rank, k_filt, 'double');
    else
      c = fwrite(randn_fid, V(:), 'double');
      assert(c == prod(size(V)));
    end
  end

  if (RANDN_CONV)
    X = PI_0_sqrt_convmtx*V;  % The initial ensemble
  else
    X = PI_0_sqrt*V;  % The initial ensemble
  end
  
  x_mean = x0;
  
  if (LENKF_DEBUG)
    fprintf(debug_fid, 'Initial ensemble:\n');
    fprintf_full(debug_fid, X);
  end

  row_H = 1;
  
  for i=1:T
    % Measurement update
    % ------------------
    ab = struct();
    ab.i = i;
    ab.H = H;
    ab.row_H = row_H;
    ab.name_H = 'H';
    ab.R_sqrt = R_sqrt;
    ab.y_vec = y(:,i);
    ab.C = C;
    ab.L = L;
    ab.LENKF_DEBUG = LENKF_DEBUG;
    if (LENKF_DEBUG)
      ab.debug_fid = debug_fid;
    end
    ab.RANDN_DEBUG = RANDN_DEBUG;
    if (RANDN_DEBUG)
      ab.randn_fid = randn_fid;
    end
    
    for row_block=1:M_block:M
      ab.x_mean = x_mean;
      ab.X = X;
      ab.row_block = row_block;
      
      if (row_block + M_block > M)
        ab.n_rows = M - row_block + 1;
      else
        ab.n_rows = M_block;
        end
      
      [x_mean, X] = compute_measurement_update(ab);
    end
    
    row_H = row_H + M;
    if (row_H > size(H,1))
      assert(row_H == size(H,1) + 1);
      row_H = 1;
    end

    if (lambda > 0)
      ab.H = D;
      ab.row_H = 1;
      ab.name_H = 'D';
      ab.R_sqrt = diag(1/sqrt(lambda)*ones(size(D,1),1));
      ab.y_vec = zeros(size(D,1), 1);
      
      for row_block=1:M_block:size(D,1)
        ab.x_mean = x_mean;
        ab.X = X;
        ab.row_block = row_block;
        
        if (row_block + M_block > size(D,1))
          ab.n_rows = size(D,1) - row_block + 1;
        else
          ab.n_rows = M_block;
        end
        
        [x_mean, X] = compute_measurement_update(ab);
      end
    end
      
    x_lenkf_f(:,i) = x_mean;

    if (0)
    if (PLOT_TRACE)
      scratch = 1/L*sum(X, 2);
      X_tilde = X - scratch*ones(1,L);
      traces(to_plot_index, i) = 1/(L-1)*X_tilde(:)'*X_tilde(:);
      X_tilde = X + scratch*ones(1,L);
      
      if (LENKF_DEBUG)
        fprintf(debug_fid, '\ntrace:\n');
        fprintf(debug_fid, '%+f\n', traces(to_plot_index, i));
      end
    end
    end
    
    % Time update
    % -----------
    
    if (RANDN_CONV)
      U = randn(J,L);
    else
      U = randn(N,L);
    end
    
    if (RANDN_DEBUG)
      if (RANDN_CONV)
        output_row_major(randn_fid, U, rank, k_filt, 'double');
      else
        c = fwrite(randn_fid, U(:), 'double');
        assert(c == prod(size(U)));
      end
    end

    % Assuming F = I
    if (RANDN_CONV)
      X = F*X + Q_sqrt_convmtx*U;
    else
      X = F*X + Q_sqrt*U;
    end
    x_mean = F*x_mean;
    
    if (LENKF_DEBUG)
      fprintf(debug_fid, '\nX:\n');
      fprintf_full(debug_fid, X);
      fprintf(debug_fid, '\nx_mean:\n');
      fprintf_full(debug_fid, x_mean');
    end
  end

  to_plot(:,:,to_plot_index) = x_lenkf_f;
  to_plot_index = to_plot_index + 1;
  to_plot_t(end+1) = {'lenkf'};

  d = x_lenkf_f - x;
  e_lenkf = norm(d(:))/norm(x(:));
  
  disp(sprintf('lenkf:\t\t%f', e_lenkf));
  
  
  if (RANDN_DEBUG)
    fclose(randn_fid);
  end
  
  if (LENKF_DEBUG)
    fclose(debug_fid);
  end
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  C implementation of the ensemble Kalman filter

if (c_lenkf)
  cfg = struct();
  cfg.cfg_filename =      '../tmp/static_2D_lenkf.cfg';
  cfg.dir_name =          '../tmp';
  cfg.suffix =            'static_2D_lenkf';
  cfg.x_hat_filename =    'x_lenkf_static_2D';
  cfg.trace_filename =    'trace_lenkf_static_2D';

  cfg.N = N;
  cfg.M_block = M_block;
  cfg.T = T;
  cfg.L = L;

  cfg.rank = rank;
  cfg.n = n_output;
  
  if (regularize)
    cfg.lambda = lambda;
  else
    cfg.lambda = 0;
  end
  
  cfg.update_epsilon = update_epsilon;

  cfg.regularize = regularize;
  cfg.randn_conv = RANDN_CONV;
  
  cfg.quiet_mode = 1;
  cfg.save_trace = PLOT_TRACE;
  cfg.randn_debug = RANDN_DEBUG;
  cfg.lenkf_debug = LENKF_DEBUG;
  
  % x0 initialized at the top
  P0_sqrt = PI_0_sqrt;
  
  MANGLE = @(x) [cfg.dir_name, '/', x, '_', cfg.suffix];
  
  export_vector(MANGLE('x0'), x0, ELEM_TYPE);
  export_rcs(MANGLE('D'), D, ELEM_TYPE);

  export_vector(sprintf('%s_0', MANGLE('y')), y, ELEM_TYPE);
  export_rcs(sprintf('%s_0', MANGLE('H')), H, ELEM_TYPE);
  export_diag(sprintf('%s_0', MANGLE('R_sqrt')), R_sqrt, ELEM_TYPE);
  
  if (RANDN_CONV)
    export_r_filter_new(MANGLE('PI_0_sqrt'), PI_0_sqrt_zp, ELEM_TYPE);
    export_r_filter_new(MANGLE('Q_sqrt'), Q_sqrt_zp, ELEM_TYPE);
  else
    export_rcs(MANGLE('PI_0_sqrt'), P0_sqrt, ELEM_TYPE);
    export_rcs(MANGLE('Q_sqrt'), Q_sqrt, ELEM_TYPE);
  end
   
  row_0 = C(1,:);
  blocks = reshape(row_0, res, res);
  export_sb_toe_r(MANGLE('C'), blocks, ELEM_TYPE);
  
  output_cfg_file_2D(cfg);

  [s,w] = system('hostname');
  assert(s == 0);
  
  prefix = get_system_prefix(ELEM_TYPE);
  cmd = sprintf('%s ../../lenkf %s', prefix, cfg.cfg_filename);

  [s, w] = system(cmd);
  assert(s == 0);
  
  x_c_lenkf_f = ...
      import_full_c(sprintf('%s/%s', cfg.dir_name, cfg.x_hat_filename), ...
                    ELEM_TYPE);

  if (0)
  if (PLOT_TRACE)
    traces(to_plot_index, :) = ...
        import_vector(sprintf('%s/%s', cfg.dir_name, cfg.trace_filename), ...
                      ELEM_TYPE);
  end
  end
  
  to_plot(:,:,to_plot_index) = x_c_lenkf_f;
  to_plot_index = to_plot_index + 1;
  to_plot_t(end+1) = {'c\_lenkf'};
  
  d = x_c_lenkf_f - x;
  e_c_lenkf = norm(d(:))/norm(x(:));
  
  disp(sprintf('c_lenkf:\t%f', e_c_lenkf));
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


CO = [0 0 0;
      0 0 1;
      1 0 0;
      0 1 0;
      1 0 1;
      0 1 1];

set(0, 'DefaultAxesColorOrder', CO);

if (0)
if (PLOT_TRACE)
figure(2);
set(gca, 'ColorOrder', CO);
plot(log10(traces'));
xlabel('i');
ylabel('log10(trace(P_{i|i}))');
legend(to_plot_t);
end
end

figure(1);
h = gcf;
for i=1:T
  if (gcf ~= h)
    figure(h);
  end
  subplot(1,N_to_plot+1,1);
  x_i = x(:,i);
  imagesc(reshape(x_i, res, res));
  axis('square');
  title(sprintf('x   %d', i));
  %colorbar;
  
  clim = [min(x_i), max(x_i)];
  
  for l=1:N_to_plot
    subplot(1,N_to_plot+1,1+l);
    imagesc(reshape(to_plot(:,i,l), res, res), clim);
    axis('square');
    title(sprintf('%s', char(to_plot_t(l))));
    %colorbar;
  end
  
  drawnow;
end
