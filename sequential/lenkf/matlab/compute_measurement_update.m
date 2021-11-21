function [x_mean, X] = compute_measurement_update(ab)

x_mean = ab.x_mean;
X = ab.X;
i = ab.i;
H = ab.H;
row_H = ab.row_H;
name_H = ab.name_H;
row_block = ab.row_block;
n_rows = ab.n_rows;
R_sqrt = ab.R_sqrt;
y_vec = ab.y_vec;
C = ab.C;
L = ab.L;
LENKF_DEBUG = ab.LENKF_DEBUG;
if (LENKF_DEBUG)
  debug_fid = ab.debug_fid;
end
RANDN_DEBUG = ab.RANDN_DEBUG;
if (RANDN_DEBUG)
  randn_fid = ab.randn_fid;
end

% No longer subtracting mean at each measurement update
%scratch = 1/L*sum(X, 2);
%X = X - scratch * ones(1,L);

H_I = row_H+(row_block-1):row_H+(row_block-1)+n_rows-1;
B_I = row_block:row_block + n_rows - 1;


if (LENKF_DEBUG)
  fprintf(debug_fid, '\n--------------------------------------------------------------------------------\n');
  fprintf(debug_fid, 'i = %d\trow_%c = %d\tn_rows = %d\n', i-1, name_H, ...
          H_I(1)-1, n_rows);
  fprintf(debug_fid, '\nx_mean:\n');
  fprintf_full(debug_fid, x_mean');
  fprintf(debug_fid, '\nX:\n');
  fprintf_full(debug_fid, X);
  fprintf(debug_fid, '\n%c(%d:%d, :):\n', name_H, H_I(1)-1, H_I(end)-1);
  fprintf_full(debug_fid, H(H_I,:));
end

P_HT = 1/(L-1)*(C .* (X*X'))*H(H_I,:)';

if (LENKF_DEBUG)
  fprintf(debug_fid, '\nP_%cT:\n', name_H);
  fprintf_full(debug_fid, P_HT(:, 1:n_rows));
end

B = H(H_I,:)*P_HT;

if (LENKF_DEBUG)
  fprintf(debug_fid, '\n%c_P_%cT:\n', name_H, name_H);
  fprintf_full(debug_fid, triu(B));
end

B = B + R_sqrt(B_I, B_I)*R_sqrt(B_I, B_I)';

if (LENKF_DEBUG)
  fprintf(debug_fid, '\n%c_P_%cT + R:\n', name_H, name_H);
  fprintf_full(debug_fid, triu(B));
end

V = randn(n_rows,L);

if (RANDN_DEBUG)
  c = fwrite(randn_fid, V(:), 'double');
  assert(c == n_rows*L);
end

E = R_sqrt(B_I, B_I)*V;

if (LENKF_DEBUG)
  fprintf(debug_fid, '\nV:\n');
  fprintf_full(debug_fid, E);
end

E = E - H(H_I,:)*X;

if (LENKF_DEBUG)
  fprintf(debug_fid, '\nV - %c_X:\n', name_H);
  fprintf_full(debug_fid, E);
end

if (LENKF_DEBUG)
  fprintf(debug_fid, '\nchol(B):\n');
  fprintf_full(debug_fid, chol(B));
end

E = B\E;

if (LENKF_DEBUG)
  fprintf(debug_fid, '\nB\\E:\n');
  fprintf_full(debug_fid, E);
end

if (LENKF_DEBUG)
  fprintf(debug_fid, '\ny:\n');
  fprintf_full(debug_fid, y_vec(B_I)');
end

scratch = y_vec(B_I) - H(H_I,:)*x_mean;

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