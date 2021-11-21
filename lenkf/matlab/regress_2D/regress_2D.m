% regress_2D.m

LIBMDB_MATRIX_PATH = '~/src/libmdb_matrix';

addpath(fullfile(LIBMDB_MATRIX_PATH, 'matlab'));

kf         = 1;
lskf       = 0;
lvkf       = 1;
lenkf      = 1;
c_lenkf    = 1;
pf         = 1;
RESAMPLE   = 1;

ELEM_TYPE = 'float64';

PLOT_TRACE = 1;

LENKF_DEBUG = 0;
LENKF_DEBUG_FILENAME= '../tmp/regress_2D.debug';

RANDN_DEBUG = 1;
RANDN_CONV  = 1;
RANDN_FILENAME = '../tmp/randn';

USE_PRIOR = 1;
NO_PRIOR_SF = 1e-12;

KF_STABLE_UPDATE = 1;

POISSON_NOISE = 0;

alpha = 1e1;
beta = 1e-1;
zeta = 1e1;

res = 4;
T = 32;
rad = 2;
d = 1/4;
v = 1/64;

regularize = 1;
lambda = 1e-2/alpha*zeta;

localize = 1;
%taper = res - 1;
taper = 0;

update_epsilon = 0;

na = res;
np = ceil(sqrt(2)*res);
d = 1;
SNR = 100;

M = np;
%M_block = res^2;
M_block = 1;
N = res^2;
%L = 2^12;
L = 256*8;

L_pf = L;

rank = 2;
n = [res, res];
n_output = [n(end:-1:3) n(1) n(2)];

corr_l = 3;
Q_sqrt_sf = 5e-2*sqrt(alpha);
PI_0_sqrt_sf = 1e0*sqrt(alpha);


if (RANDN_DEBUG || LENKF_DEBUG)
  rand('seed', 1);
  randn('seed', 1);
end


% Creates y and H
setup_regress_2D;

F = speye(N,N);

R = var_v*speye(M,M);
R_sqrt = sig_v*speye(M,M);

na = min([na, T]);
y_na = y(:, 1:na);
y_na = y_na(:);
x0 = 4*H(1:np*na, :)' * y_na / beta^2;

%x0 = zeros(N,1);


if (regularize)
  [Dx, Dy] = dgrad([res, res]);
  D = [Dx; Dy];
  D = D / sqrt(svds(Dx,1)^2 + svds(Dy,1)^2);
else
  D = zeros(0);
end
[D_M, D_N] = size(D);
D = D / sqrt(zeta);

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


N_to_plot = sum([kf lskf lvkf lenkf c_lenkf pf]);

to_plot = zeros(N,T,N_to_plot);
to_plot_t = {};
to_plot_index = 1;

traces = zeros(N_to_plot, T);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  Kalman filter

if (kf)
  row_H = 1;

  x_kf_f = zeros(N,T);   % The filtered estimate  x_{i|i}
  x_kf_p = zeros(N,T+1); % The predicted estimate x_{i|i-1}

  P_p = zeros(N,N);      % The predicted error covariance P_{i|i-1}
  P_f = zeros(N,N);      % The filtered error covariance  P_{i|i}

  if (USE_PRIOR)
      x_kf_p(:,1) = x0;
      P_p = PI_0;  % Initial state covariance
  end

  for i=1:T
    H_I = row_H:row_H+M-1;

    if (i == 1 && ~USE_PRIOR)
        P_f = inv(H(H_I,:)'*inv(R)*H(H_I,:) + NO_PRIOR_SF*speye(N));
        x_kf_f(:,i) = P_f*H(H_I,:)'*inv(R)*y(:,i);
    else
        % Measurement update
        K = P_p*H(H_I,:)'*inv(H(H_I,:)*P_p*H(H_I,:)' + R);
        x_kf_f(:,i) = x_kf_p(:,i) + K*(y(:,i) - H(H_I,:)*x_kf_p(:,i));

        if (KF_STABLE_UPDATE)
            P_f = P_p - K*H(H_I,:)*P_p - P_p*H(H_I,:)'*K' + ...
                  K*(H(H_I,:)*P_p*H(H_I,:)' + R)*K';
        else
            P_f = P_p - K*H(H_I,:)*P_p;
        end
    end

    row_H = row_H + M;
    if (row_H > size(H,1))
      row_H = 1;
    end

    if (regularize)
      K = P_f*D'*inv(D*P_f*D' + 1/lambda*eye(D_M));
      x_kf_f(:,i) = x_kf_f(:,i) - K*D*x_kf_f(:,i);
      if (KF_STABLE_UPDATE)
          P_f = P_f - K*D*P_f - P_f*D'*K' + K*(D*P_f*D' + 1/lambda*eye(D_M))*K';
      else
          P_f = P_f - K*D*P_f;
      end
    end

    if (PLOT_TRACE)
      traces(to_plot_index, i) = trace(P_f);
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

    if (PLOT_TRACE)
      traces(to_plot_index, i) = trace(P_f);
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

  P_p = zeros(N,N);      % The predicted error covariance P_{i|i-1}
  P_f = zeros(N,N);      % The filtered error covariance  P_{i|i}

  if (USE_PRIOR)
      x_lvkf_p(:,1) = x0;
      P_p = PI_0;
  end

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

      if (i == 1 && m == 1 && ~USE_PRIOR)
          P_f = inv(H(H_I,:)'*inv(R(B_I, B_I))*H(H_I,:) + NO_PRIOR_SF*speye(N));
          x_lvkf_f(:,i) = P_f*H(H_I,:)'*inv(R(B_I, B_I))*y(B_I,i);
      else
          K = (C.*P_f)*H(H_I,:)'...
              *inv(H(H_I,:)*(C.*P_f)*H(H_I,:)' + R(B_I, B_I));
          x_lvkf_f(:,i) = x_lvkf_f(:,i) + K*(y(B_I,i) - H(H_I,:)*x_lvkf_f(:,i));
          P_f = P_f - K*H(H_I,:)*P_f - P_f*H(H_I,:)'*K' + ...
                K*(H(H_I,:)*P_f*H(H_I,:)' + R(B_I,B_I))*K';
      end

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

    if (PLOT_TRACE)
      traces(to_plot_index, i) = trace(P_f);
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
  suffix =            'regress_2D_lenkf';

  cfg = struct();
  cfg.cfg_filename =      '../tmp/regress_2D_lenkf.cfg';
  cfg.dir_name =          '../tmp';

  MANGLE = @(z) [cfg.dir_name, '/', z, '_', suffix];

  cfg.x_hat_filename =    MANGLE('x_regress_2D_lenkf');
  cfg.trace_filename =    MANGLE('trace_regress_2D_lenkf');

  cfg.x0_filename =       MANGLE('x0');
  cfg.PI_sqrt_filename =  MANGLE('PI_0_sqrt');
  cfg.D_filename =        MANGLE('D');
  cfg.Q_sqrt_filename =   MANGLE('Q_sqrt');
  cfg.C_filename =        MANGLE('C');

  cfg.y_list_filename =   MANGLE('y_list');
  cfg.H_list_filename =   MANGLE('H_list');
  cfg.R_sqrt_list_filename = MANGLE('R_sqrt_list');


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

  % Remove all symlinks from previous experiment
  cmd = sprintf('find ../tmp/*%s* -type l | xargs rm -f', suffix);
  [s, w] = system(cmd);
  assert(s == 0);

  export_vector(cfg.x0_filename, x0, ELEM_TYPE);
  export_rcs(cfg.D_filename, D, ELEM_TYPE);

  y_fid = fopen(cfg.y_list_filename, 'w');
  assert(y_fid ~= -1);

  H_fid = fopen(cfg.H_list_filename, 'w');
  assert(H_fid ~= -1);

  R_sqrt_fid = fopen(cfg.R_sqrt_list_filename, 'w');
  assert(R_sqrt_fid ~= -1);

  for i=1:T
      R_sqrt_fname = sprintf('%s_%d', MANGLE('R_sqrt'), i-1);
    if (i == 1)
      export_diag(R_sqrt_fname, R_sqrt, ELEM_TYPE);
    else
      cmd = sprintf('ln -sf %s_0 %s', MANGLE('R_sqrt'), R_sqrt_fname);
      [s, w] = system(cmd);
      assert(s == 0);
    end

    y_fname = sprintf('%s_%d', MANGLE('y'), i-1);
    H_fname = sprintf('%s_%d', MANGLE('H'), i-1);

    if (i <= na)
      I = ((i-1)*M+1):(i*M);
      export_rcs(H_fname, H(I,:), ELEM_TYPE);
    else
      cmd = sprintf('ln -sf %s_%d %s_%d', MANGLE('H'), mod(i-1, na), ...
                    MANGLE('H'), i-1);
      [s, w] = system(cmd);
      assert(s == 0);
    end

    export_vector(y_fname, y(:,i), ELEM_TYPE);

    fprintf(y_fid, '%s\n', y_fname);
    fprintf(H_fid, '%s\n', H_fname);
    fprintf(R_sqrt_fid, '%s\n', R_sqrt_fname);
  end

  fclose(y_fid);
  fclose(H_fid);
  fclose(R_sqrt_fid);

  if (RANDN_CONV)
    export_r_filter_new(cfg.PI_sqrt_filename, PI_0_sqrt_zp, ELEM_TYPE);
    export_r_filter_new(cfg.Q_sqrt_filename, Q_sqrt_zp, ELEM_TYPE);
  else
    export_rcs(cfg.PI_sqrt_filename, PI_0_sqrt, ELEM_TYPE);
    export_rcs(cfg.Q_sqrt_filename, Q_sqrt, ELEM_TYPE);
  end

  row_0 = C(1,:);
  blocks = reshape(row_0, res, res);
  export_sb_toe_r(cfg.C_filename, blocks, ELEM_TYPE);

  output_cfg_file_2D(cfg);

  cmd = sprintf('../../lenkf %s', cfg.cfg_filename);

  [s, w] = system(cmd);
  assert(s == 0);

  x_c_lenkf_f = ...
      import_full_c(cfg.x_hat_filename);

  if (PLOT_TRACE)
    traces(to_plot_index, :) = ...
        import_vector(cfg.trace_filename);
  end

  to_plot(:,:,to_plot_index) = x_c_lenkf_f;
  to_plot_index = to_plot_index + 1;
  to_plot_t(end+1) = {'c\_lenkf'};

  d = x_c_lenkf_f - x;
  e_c_lenkf = norm(d(:))/norm(x(:));

  disp(sprintf('c_lenkf:\t%f', e_c_lenkf));
end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  Particle filter (with optimal proprosal distribution)

if (pf)
  row_H = 1;

  x_pf_f = zeros(N,T);   % The filtered estimate  x_{i|i}

  %x_pf = x0*ones(1, L_pf);
  x_pf = x0*ones(1, L_pf) + PI_0_sqrt*randn(N, L_pf);

  w = ones(L_pf, 1);
  N_eff = zeros(T, 1);

  for i=1:T
    H_I = row_H:row_H+M-1;

    if (regularize)
      H_p = [H(H_I,:);
             D];
      R_p = blkdiag(R, 1/lambda*eye(D_M));
      y_p = [y(:, i);
             zeros(D_M, 1)];
    else
      H_p = H(H_I,:);
      R_p = R;
      y_p = y(:, i);
    end

    % Update importance weights
    A_pf = (-1/2)*(y_p*ones(1,L_pf) - H_p*F*x_pf)'*inv(R_p + H_p*Q*H_p')*(y_p*ones(1,L_pf) - H_p*F*x_pf);
    p_l = exp(diag(A_pf));
    w = w .* p_l;

    % Find normalized weights
    %w_norm = w / sum(w);
    w = w / sum(w);
    w_norm = w;
    N_eff(i) = 1/sum(w_norm.^2);

    % Draw samples from the proposal
    Sigma = inv(inv(Q) + H_p'*inv(R_p)*H_p);
    Sigma_sqrt = chol(Sigma)';
    U = Sigma_sqrt*randn(N, L_pf);

    m = Sigma*(inv(Q)*F*x_pf + H_p'*inv(R_p)*y_p*ones(1,L_pf));
    x_pf = m + U;

    row_H = row_H + M;
    if (row_H > size(H,1))
      row_H = 1;
    end

    % Compute filtered estimate
    x_pf_f(:, i) = sum(x_pf*diag(w_norm), 2);

    % Resample
    if (RESAMPLE)
      x_pf_old = x_pf;
      w_cdf = cumsum(w);

      I = zeros(L_pf, 1);

      for l=1:L_pf
        I(l) = find(w_cdf >= rand(1), 1);
      end

      x_pf = x_pf_old(:, I);

      w = ones(size(w));
    end
  end

  to_plot(:,:,to_plot_index) = x_pf_f;
  to_plot_index = to_plot_index + 1;
  to_plot_t(end+1) = {'pf'};

  d = x_pf_f - x;
  e_pf = norm(d(:))/norm(x(:));

  disp(sprintf('pf:\t\t%f', e_pf));
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


CO = [0 0 0;
      0 0 1;
      1 0 0;
      0 1 0;
      1 0 1;
      0 1 1];

set(0, 'DefaultAxesColorOrder', CO);

if (PLOT_TRACE)
figure(2);
set(gca, 'ColorOrder', CO);
plot(log10(traces'));
xlabel('i');
ylabel('log10(trace(P_{i|i}))');
legend(to_plot_t);
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
    axis('image');
    title(sprintf('%s', char(to_plot_t(l))));
    %colorbar;
  end

  drawnow;
  pause;
end
