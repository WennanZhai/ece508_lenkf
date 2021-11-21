% regress_1D.m

% Spatial regularization is not required in this example --- the prior x0=0 with
% small enough initial error covariance is sufficient to stabilize the problem.

T = 32;              % number of time steps
N = 8;               % size of state vector
M = 7;               % number of rows in measurement operator
                     %L = 2^10;            % number of members of the ensemble
L = 2^10;
L_pf = L;


kf          = 1;     % Kalman filter?
tkf         = 1;     % Toeplitz Kalman filter?
lkf         = 1;     % localized Kalman filter?
lsenkf      = 0;     % localized serial ensemble KF?
lenkf       = 1;     % localized ensemble Kalman filter?
c_lenkf     = 1;     % C implementation of the lenkf?

pf          = 1;
RESAMPLE    = 1;

localize    = 1;
%taper       = N - 1; % # of neighbors to include in the mask
taper = 2;

ELEM_TYPE = 'float64';

PLOT_TRACE  = 1;

LENKF_DEBUG = 0;
LENKF_DEBUG_FILENAME= '../tmp/regress_1D.debug';

RANDN_DEBUG = 1;
RANDN_FILENAME = '../tmp/randn';

POISSON_NOISE = 0;
POISSON_EPS = 1e1;  % Smallest value allowed on the diagonal of R - must be
                    % greater than 0 or there will be numerical issues -
                    % linked with alpha below (they are proportional)

alpha = 1e6;
beta = 1e1;

LIBMDB_MATRIX_PATH = '~/src/libmdb_matrix';


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


addpath('../');
addpath(fullfile(LIBMDB_MATRIX_PATH, 'matlab'));

if (RANDN_DEBUG)
  rand('seed', 10);
  randn('seed', 10);
end


% random spot measurement operator
n_spots = 1;  % number of spots in random_spot matrix

H = zeros(M,N,T);
for t = 1:T
  for m = 1:M
    for nss = 1:n_spots
      [mm,ii] = max(rand(1,N));
      H(m,ii,t) = 1.0;
    end
  end
  s = svd(H(:,:,t));
  meig = max(s);
  H(:,:,t) = H(:,:,t)/meig;
end

H = H*beta;

%%% IGNORED IF POISSON NOISE IS SET
% measurement covariance
sigR  = 2e-2;  % measurement covariance scaling
R_sqrt = eye(M,M);
R = R_sqrt*R_sqrt';
meig = max(eig(R));
R = R*sigR^2/meig*alpha*beta^2;
R_sqrt = R_sqrt*sigR/sqrt(meig)*sqrt(alpha)*beta;

% state covariance model (the true state dynamics are
%   deterministic)
sigQ = 1e-1; % state covariane scaling
corr_l = 1;
K_log = 2*corr_l;
Q_h = triang(K_log);
M_h = N + K_log - 1;
if (mod(M_h,2) == 1)
  M_h = M_h + 1;
end
q = conv(Q_h, Q_h)';
Q = toeplitz([q(K_log:end), zeros(1, N - K_log)]);
Q_sqrt = chol(Q)';
meig = max(eig(Q));
Q_norm = Q/meig;
Q_sqrt_norm = Q_sqrt/sqrt(meig);
Q = Q_norm*sigQ^2*alpha;
Q_sqrt = Q_sqrt_norm*sigQ*sqrt(alpha);
%ALPHA? Q_h = Q_h*sigQ/sqrt(meig);
%ALPHA? Q_h_pad = [Q_h(K_log/2+1:end); zeros(M_h-K_log,1); Q_h(1:K_log/2)];
%ALPHA? Q_H_pad = fft(Q_h_pad);


x0 = zeros(N,1);
%x0 = ones(N,1)*sqrt(alpha);

PI_0_sf = 1e-1;

PI_0 = PI_0_sf*Q_norm*alpha;
PI_0_sqrt = sqrt(PI_0_sf)*Q_sqrt_norm*sqrt(alpha);

F = eye(N,N);

lx = linspace(0,1,N)';  % position grid
t  = linspace(0,1,T);   % time grid
center = 0.5 + 0.35*sin(2*pi*t); % center of harmonic oscillator

% each column of x is a state vector
%                y      meaurement
x = zeros(N,T);
y = zeros(M,T);
y_true = zeros(M,T);
n = zeros(M,T);

for k = 1:T
  x(:,k) = sqrt(alpha)*exp(-(lx - center(k)).^2/0.2^2);
  if (POISSON_NOISE)
    y_true(:,k) = H(:,:,k)*x(:,k);
    y(:,k) = poissrnd(y_true(:,k));
    n(:,k) = y(:,k) - y_true(:,k);
  else
    y_true(:,k) = H(:,:,k)*x(:,k);
    n(:,k) = R_sqrt*randn(M,1);
    y(:,k) = y_true(:,k) + n(:,k);
  end
end

to_plot = [x];


n_to_save = sum([kf, tkf, lkf, lsenkf, lenkf]);
traces = zeros(n_to_save, T);
traces_names = {};
trace_index = 1;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  Kalman filter

if (kf)
  x_kf_f = zeros(N,T);   % The filtered estimate  x_{i|i}
  x_kf_p = zeros(N,T+1); % The predicted estimate x_{i|i-1}
  x_kf_p(:,1) = x0;

  P_p = zeros(N,N);      % The predicted error covariance P_{i|i-1}
  P_f = zeros(N,N);      % The filtered error covariance  P_{i|i}

  P_p = PI_0;

  for i=1:T
    if (POISSON_NOISE)
      R = diag(max(y(:, i), POISSON_EPS^2));
    end

    % Measurement update
    K = P_p*H(:,:,i)'*inv(H(:,:,i)*P_p*H(:,:,i)' + R);
    x_kf_f(:,i) = x_kf_p(:,i) + K*(y(:,i) - H(:,:,i)*x_kf_p(:,i));
    P_f = P_p - K*H(:,:,i)*P_p;

    if (i == 1)
      K_1 = K;
      e_1 = y(:,i) - H(:,:,i)*x_kf_p(:,i);
      y_1 = y(:,i);
      z_1 = H(:,:,i)*x_kf_p(:,i);
    end

    if (PLOT_TRACE)
      traces(trace_index, i) = trace(P_f);
    end

    % Time update
    x_kf_p(:,i+1) = F*x_kf_f(:,i);
    P_p = F*P_f*F' + Q;
  end

  to_plot = [to_plot; x_kf_f];

  d = x_kf_f - x;
  e_kf = norm(d(:))/norm(x(:));

  disp(sprintf('kf:\t\t%f', e_kf));

  if (PLOT_TRACE)
    traces_names(end+1) = {'kf'};
    trace_index = trace_index + 1;
  end
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  Kalman filter

if (tkf)
  x_tkf_f = zeros(N,T);   % The filtered estimate  x_{i|i}
  x_tkf_p = zeros(N,T+1); % The predicted estimate x_{i|i-1}
  x_tkf_p(:,1) = x0;

  P_p = zeros(N,N);      % The predicted error covariance P_{i|i-1}
  P_f = zeros(N,N);      % The filtered error covariance  P_{i|i}

  P_p = toeplitzify(PI_0);

  for i=1:T
    % Measurement update
    K = P_p*H(:,:,i)'*inv(H(:,:,i)*P_p*H(:,:,i)' + R);
    x_tkf_f(:,i) = x_tkf_p(:,i) + K*(y(:,i) - H(:,:,i)*x_tkf_p(:,i));
    P_f = toeplitzify(P_p - K*H(:,:,i)*P_p);

    if (PLOT_TRACE)
      traces(trace_index, i) = trace(P_f);
    end

    % Time update
    x_tkf_p(:,i+1) = F*x_tkf_f(:,i);
    P_p = toeplitzify(F*P_f*F' + Q);
  end

  to_plot = [to_plot; x_tkf_f];

  d = x_tkf_f - x;
  e_tkf = norm(d(:))/norm(x(:));

  disp(sprintf('tkf:\t\t%f', e_tkf));

  if (PLOT_TRACE)
    traces_names(end+1) = {'tkf'};
    trace_index = trace_index + 1;
  end
end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  Generate G&C localization mask

if (localize)
  C = gen_gc_mask_1D(N, taper);
else
  C = ones(N, N);
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  Localized Kalman filter

if (lkf)
  x_lkf_f = zeros(N,T);   % The filtered estimate  x_{i|i}
  x_lkf_p = zeros(N,T+1); % The predicted estimate x_{i|i-1}
  x_lkf_p(:,1) = x0;

  P_p = zeros(N,N);      % The predicted error covariance P_{i|i-1}
  P_f = zeros(N,N);      % The filtered error covariance  P_{i|i}

  P_p = PI_0;

  for i=1:T
    % Measurement update
    K = (C.*P_p)*H(:,:,i)'*inv(H(:,:,i)*(C.*P_p)*H(:,:,i)' + R);
    x_lkf_f(:,i) = x_lkf_p(:,i) + K*(y(:,i) - H(:,:,i)*x_lkf_p(:,i));
    P_f = P_p - K*H(:,:,i)*P_p - P_p*H(:,:,i)'*K' + ...
          K*(H(:,:,i)*P_p*H(:,:,i)' + R)*K';

    if (PLOT_TRACE)
      traces(trace_index, i) = trace(P_f);
    end

    % Time update
    x_lkf_p(:,i+1) = F*x_lkf_f(:,i);
    P_p = F*P_f*F' + Q;
  end

  to_plot = [to_plot; x_lkf_f];

  d = x_lkf_f - x;
  e_lkf = norm(d(:))/norm(x(:));

  disp(sprintf('lkf:\t\t%f', e_lkf));

  if (PLOT_TRACE)
    traces_names(end+1) = {'lkf'};
    trace_index = trace_index + 1;
  end
end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  Ensemble Kalman filter

if (lsenkf)
  % x0 initialized at the top
  P0_sqrt = PI_0_sqrt;

  e = zeros(N, L);         % The ensemble
  x_lsenkf_f = zeros(N, T);  % The filtered estimate x_{i|i}

  % Initialize ensemble
  r = randn(N,L);
  e = x0 * ones(1, L) + P0_sqrt*r;

  for i=1:T
    % Measurement update

    x_bar = 1/L*sum(e,2);
    e_tilde = e - x_bar*ones(1,L);

    for m=1:M
      % mth measurement update
      P_hT = zeros(N, 1);
      for n=1:N
        d_n = (C(n,:).*H(m,:,i))*e_tilde;
        P_hT(n) = 1/(L-1)*e_tilde(n,:)*d_n';
      end

      h_P_hT = H(m,:,i)*P_hT;

      k = P_hT/(h_P_hT + R(m,m));

      x_bar = x_bar + k*(y(m,i) - H(m,:,i)*x_bar);

      for l=1:L
        n = randn(1,1);
        y_l = R_sqrt(m,m)*n;
        e_tilde(:,l) = e_tilde(:,l) + k*(y_l - H(m,:,i)*e_tilde(:,l));
      end
    end

    if (PLOT_TRACE)
      traces(trace_index, i) = 1/(L-1)*e_tilde(:)'*e_tilde(:);
    end

    x_lsenkf_f(:,i) = x_bar;

    e = e_tilde + x_bar*ones(1,L);

    % Time update
    if (0)
    n = randn(M_h,L);

    for l=1:L
      u_l = ifft(Q_H_pad .* fft(n(:,l)));
      u_l = u_l(1:N);
      e(:,l) = F*e(:,l) + u_l;
    end
    end

    n = randn(N,L);

    e = F*e + Q_sqrt*n;
  end

  to_plot = [to_plot; x_lsenkf_f];

  d = x_lsenkf_f - x;
  e_lsenkf = norm(d(:))/norm(x(:));

  disp(sprintf('lsenkf:\t\t%f', e_lsenkf));

  if (PLOT_TRACE)
    traces_names(end+1) = {'lsenkf'};
    trace_index = trace_index + 1;
  end
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  Localized ensemble Kalman filter

if (lenkf)
  if (c_lenkf && RANDN_DEBUG)
    randn_fid = my_fopen(RANDN_FILENAME, 'w');
  end

  if (LENKF_DEBUG)
    debug_fid = fopen(LENKF_DEBUG_FILENAME, 'w');
  end

  x_lenkf_f = zeros(N,T);   % The filtered estimate  x_{i|i}

  n = randn(N,L);
  if (c_lenkf && RANDN_DEBUG)
    c = fwrite(randn_fid, n(:), 'double');
    assert(c == N*L);
  end

  X = PI_0_sqrt*n;  % The initial ensemble
  x_mean = x0;

  if (LENKF_DEBUG)
    fprintf(debug_fid, 'Initial ensemble:\n');
    fprintf_full(debug_fid, X);
  end

  for i=1:T
    % Measurement update
    % ------------------

    scratch = 1/L*sum(X, 2);
    %X = X - scratch * ones(1,L);

    if (LENKF_DEBUG)
      fprintf(debug_fid, '\n--------------------------------------------------------------------------------\n');
      fprintf(debug_fid, 'i = %d\n', i-1);
      fprintf(debug_fid, '\nx_mean:\n');
      fprintf_full(debug_fid, x_mean');
      fprintf(debug_fid, '\nX:\n');
      fprintf_full(debug_fid, X);
      fprintf(debug_fid, '\nH:\n');
      fprintf_full(debug_fid, H(:,:,i));
    end

    P_HT = 1/(L-1)*(C .* (X*X'))*H(:,:,i)';

    if (LENKF_DEBUG)
      fprintf(debug_fid, '\nP_HT:\n');
      fprintf_full(debug_fid, P_HT);
    end

    B = H(:,:,i)*P_HT;

    if (LENKF_DEBUG)
      fprintf(debug_fid, '\nH_P_HT:\n');
      fprintf_full(debug_fid, tril(B));
    end

    if (POISSON_NOISE)
      R = diag(max(y(:, i), POISSON_EPS^2));
    end

    B = B + R;

    if (LENKF_DEBUG)
      fprintf(debug_fid, '\nH_P_HT + R:\n');
      fprintf_full(debug_fid, tril(B));
    end

    V = randn(M,L);

    if (c_lenkf && RANDN_DEBUG)
        c = fwrite(randn_fid, V(:), 'double');
        assert(c == M*L);
    end

    if (POISSON_NOISE)
      R_sqrt = diag(max(sqrt(y(:, i)), POISSON_EPS));
    end

    E = R_sqrt*V;

    if (LENKF_DEBUG)
      fprintf(debug_fid, '\nV:\n');
      fprintf_full(debug_fid, E);
    end

    E = E - H(:,:,i)*X;

    if (LENKF_DEBUG)
      fprintf(debug_fid, '\nV - H_X:\n');
      fprintf_full(debug_fid, E);
    end

    if (LENKF_DEBUG)
        fprintf(debug_fid, '\nchol(B):\n');
        fprintf_full(debug_fid, chol(B)');
    end

    E = B\E;

    if (LENKF_DEBUG)
      fprintf(debug_fid, '\nB\\E:\n');
      fprintf_full(debug_fid, E);
    end

    if (LENKF_DEBUG)
      fprintf(debug_fid, '\ny:\n');
      fprintf_full(debug_fid, y(:, i)');
    end

    scratch = y(:, i) - H(:,:,i)*x_mean;

    if (LENKF_DEBUG)
      fprintf(debug_fid, '\ne:\n');
      fprintf_full(debug_fid, scratch');
    end

    scratch = B\scratch;

    if (LENKF_DEBUG)
      fprintf(debug_fid, '\nB\\e:\n');
      fprintf_full(debug_fid, scratch');
    end

    X = X + P_HT*E;

    if (LENKF_DEBUG)
        fprintf(debug_fid, '\nX:\n');
        fprintf_full(debug_fid, X);
    end

    x_mean = x_mean + P_HT*scratch;

    if (LENKF_DEBUG)
        fprintf(debug_fid, '\nx_mean:\n');
        fprintf_full(debug_fid, x_mean');
    end

    x_lenkf_f(:,i) = x_mean;

    if (PLOT_TRACE)
      scratch = 1/L*sum(X, 2);
      X_tilde = X - scratch*ones(1,L);
      traces(trace_index, i) = 1/(L-1)*X_tilde(:)'*X_tilde(:);
      X_tilde = X + scratch*ones(1,L);

      if (LENKF_DEBUG)
        fprintf(debug_fid, '\ntrace:\n');
        fprintf_full(debug_fid, traces(trace_index, i));
      end
    end

    % Time update
    % -----------

    U = randn(N,L);

    if (c_lenkf && RANDN_DEBUG)
        c = fwrite(randn_fid, U(:), 'double');
        assert(c == N*L);
    end

    % Assuming F = I
    X = F*X + Q_sqrt*U;
    x_mean = F*x_mean;

    if (LENKF_DEBUG)
        fprintf(debug_fid, '\nX:\n');
        fprintf_full(debug_fid, X);
        fprintf(debug_fid, '\nx_mean:\n');
        fprintf_full(debug_fid, x_mean');
    end
  end

  to_plot = [to_plot; x_lenkf_f];

  d = x_lenkf_f - x;
  e_lenkf = norm(d(:))/norm(x(:));

  disp(sprintf('lenkf:\t\t%f', e_lenkf));

  if (PLOT_TRACE)
    traces_names(end+1) = {'lenkf'};
    trace_index = trace_index + 1;
  end

  if (c_lenkf && RANDN_DEBUG)
    fclose(randn_fid);
  end

  if (LENKF_DEBUG)
    fclose(debug_fid);
  end
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  C implementation of the serial ensemble Kalman filter

if (c_lenkf)
  suffix =                'regress_1D_lenkf';

  cfg = struct();
  cfg.cfg_filename =      '../tmp/regress_1D_lenkf.cfg';
  cfg.dir_name =          '../tmp';

  MANGLE = @(x) [cfg.dir_name, '/', x, '_', suffix];

  cfg.x_hat_filename =    MANGLE('x_lenkf_regress_1D_lenkf');
  cfg.trace_filename =    MANGLE('trace_lenkf_regress_1D_lenkf');

  cfg.x0_filename =       MANGLE('x0');
  cfg.PI_sqrt_filename =  MANGLE('PI_0_sqrt');
  cfg.D_filename =        MANGLE('D');
  cfg.Q_sqrt_filename =   MANGLE('Q_sqrt');
  cfg.C_filename =        MANGLE('C');

  cfg.y_list_filename =   MANGLE('y_list');
  cfg.H_list_filename =   MANGLE('H_list');
  cfg.R_sqrt_list_filename = MANGLE('R_sqrt_list');

  cfg.rank = 1;
  cfg.n = N;

  cfg.N = N;
  cfg.T = T;
  cfg.L = L;
  cfg.M_block = min(M, N);

  cfg.quiet_mode = 1;
  cfg.save_trace = PLOT_TRACE;
  cfg.lambda = 0;
  cfg.poisson_noise = POISSON_NOISE;
  cfg.poisson_eps = POISSON_EPS;
  cfg.randn_debug = RANDN_DEBUG;
  cfg.lenkf_debug = LENKF_DEBUG;

  % x0 initialized at the top
  P0_sqrt = PI_0_sqrt;

  D = zeros(0);

  % Remove all symlinks from previous experiment
  cmd = sprintf('find ../tmp/*%s* -type l | xargs rm -f', suffix);
  [s, w] = system(cmd);
  assert(s == 0);

  export_vector(cfg.x0_filename, x0, ELEM_TYPE);
  export_rcs(cfg.PI_sqrt_filename, P0_sqrt, ELEM_TYPE);
  export_rcs(cfg.D_filename, D, ELEM_TYPE);
  export_rcs(cfg.Q_sqrt_filename, Q_sqrt, ELEM_TYPE);

  y_fid = fopen(cfg.y_list_filename, 'w');
  assert(y_fid ~= -1);

  H_fid = fopen(cfg.H_list_filename, 'w');
  assert(H_fid ~= -1);

  R_sqrt_fid = fopen(cfg.R_sqrt_list_filename, 'w');
  assert(R_sqrt_fid ~= -1);

  for i=1:T
    if (~POISSON_NOISE)
      R_sqrt_fname = sprintf('%s_%d', MANGLE('R_sqrt'), i-1);
      if (i == 1)
        export_diag(R_sqrt_fname, R_sqrt, ELEM_TYPE);
      else
        cmd = sprintf('ln -sf %s_0 %s', MANGLE('R_sqrt'), ...
                      R_sqrt_fname);
        [s, w] = system(cmd);
        assert(s == 0);
      end
    end

    y_fname = sprintf('%s_%d', MANGLE('y'), i-1);
    H_fname = sprintf('%s_%d', MANGLE('H'), i-1);

    export_vector(y_fname, y(:,i), ELEM_TYPE);
    export_rcs(H_fname, H(:,:,i), ELEM_TYPE);

    fprintf(y_fid, '%s\n', y_fname);
    fprintf(H_fid, '%s\n', H_fname);
    fprintf(R_sqrt_fid, '%s\n', R_sqrt_fname);
  end

  fclose(y_fid);
  fclose(H_fid);
  fclose(R_sqrt_fid);

  row_0 = C(1,:);
  export_sb_toe_r(cfg.C_filename, row_0, ELEM_TYPE);

  output_cfg_file_1D(cfg);

  cmd = sprintf('../../lenkf %s', cfg.cfg_filename);

  [s, w] = system(cmd);
  assert(s == 0);

  x_c_lenkf_f = import_full_c(cfg.x_hat_filename);

  to_plot = [to_plot; x_c_lenkf_f];

  d = x_c_lenkf_f - x;
  e_c_lenkf = norm(d(:))/norm(x(:));

  disp(sprintf('c_lenkf:\t%f', e_c_lenkf));

  if (PLOT_TRACE)
    traces(trace_index, :) = ...
        import_vector(cfg.trace_filename);
    traces_names(end+1) = {'c\_lenkf'};
    trace_index = trace_index + 1;
  end
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Particle filter (with optimal proprosal distribution)

if (pf)
  x_pf_f = zeros(N,T);   % The filtered estimate  x_{i|i}

  x_pf = x0*ones(1, L_pf) + PI_0_sqrt*randn(N, L_pf);

  w = ones(L_pf, 1);
  N_eff = zeros(T, 1);

  for i=1:T
    % Update importance weights
    A = (-1/2)*(y(:,i)*ones(1,L_pf) - H(:,:,i)*F*x_pf)'*inv(R + H(:,:,i)*Q*H(:,:,i)')*(y(:,i)*ones(1,L_pf) - H(:,:,i)*F*x_pf);
    p_l = exp(diag(A));
    w = w .* p_l;

    % Find normalized weights
    %w_norm = w / sum(w);
    w = w / sum(w);
    w_norm = w;
    N_eff(i) = 1/sum(w_norm.^2);

    % Draw samples from the proposal
    Sigma = inv(inv(Q) + H(:,:,i)'*inv(R)*H(:,:,i));
    Sigma_sqrt = chol(Sigma)';
    U = Sigma_sqrt*randn(N, L_pf);

    m = Sigma*(inv(Q)*F*x_pf + H(:,:,i)'*inv(R)*y(:,i)*ones(1,L_pf));
    x_pf = m + U;

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

  to_plot = [to_plot; x_pf_f];

  d = x_pf_f - x;
  e_pf = norm(d(:))/norm(x(:));

  disp(sprintf('pf:\t\t%f', e_pf));
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



figure(1);
imagesc(to_plot);
colorbar();


if (PLOT_TRACE)
if (prod(size(traces)) > 0)
  figure(2);
  plot(traces');
  xlabel('i');
  ylabel('trace(P_{i|i})');
  legend(traces_names);
end
end
