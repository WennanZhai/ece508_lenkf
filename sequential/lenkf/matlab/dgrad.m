function [Dx,Dy]=dgrad(dims,shape)
% [Dx,Dy]=dgrad(dims,'shape')
%
% Dx,Dy : Discrete approximations to the x and y gradient operators. 
% dims  : Vector of dimensions of the associated discrete field,
%         dims=[n1,n2] for an (n1 X n2) field. 
% shape : OPTIONAL. String specificying the shape of the resulting
%         derivative matrices. In general a discrete derivative results in a
%         row or column dimension one lower than the original. By
%         setting this parameter to shape='same', matrices are produced that
%         retain the original shape. i.e. Dx and Dy will be square. By
%         setting this parameter to shape='small' a non-square pair of
%         matrices is produced.
%         DEFAULT: shape = 'small'
%
% If F is an (n1 X n2) 2D field and f=F(:) is its vector representation,
% then Dx*f is a discrete approximation of the x gradient and Dy*f is a
% discrete approximation of the y gradient. In particular:
%
%      [-I  I     ]         [D       ]       [-1  1     ]
%      [  \  \    ]         [  \     ]       [  \  \    ]
% Dx = [   \  \   ],   Dy = [   \    ],  D = [   \  \   ]
%      [    \  \  ]         [    \   ]       [    \  \  ]
%      [    -I  I ]         [      D ]       [    -1  1 ]
% 
% where I is a n1 X n1 identity matrix. Dx is (n1*n2-n1 X n1*n2) and Dy
% is (n1*n2-n2 X n1*n2). Dx and Dy are generated as sparse matrices. Note:
% If the field is a column vector (n1 X 1) then Dx will be empty and Dy will
% be the 1-D derivative along the column. Conversely, if the field is a row
% vector (1 X n2) then Dy will be empty and Dx will be the 1-D derative
% across the row. Note that this is the logical behavior. 

% W. C. Karl 4/98

if nargin < 2
  shape = 'small';
end;

% Get dimensions
n1 = dims(1);    % y dimension of field
n2 = dims(2);    % x dimension of field
n1Dx = n1*n2-n1; % y dimension of Dx
n1Dy = n1*n2-n2; % y dimension of Dy
n2D = n1*n2;     % x dimension of Dx and Dy

% First Decide Shape
if strcmp(shape,'same') % Shape same. i.e. Dx,Dy should be square matrices

  % Generate Dx matrix
  if n1Dx > 0
    e = ones(n2,1);                    % e is an n2 vector of ones
    T = spdiags([-e e],[0 1],n2-1,n2); % T gives pattern of -I,I blocks in Dx
    T = [T;T(n2-1,:)];                 % Replicates last block of Dx
    E = speye(n1);                     % E is n1 X n1 identity matrix
    Dx = kron(T,E);                    % Puts an E block at each +-1 of T
  else
    Dx = [];
  end
 
  % Generate Dy matrix
  if n1Dy > 0
    e = ones(n1,1);                    % e is n1 vector of ones
    D = spdiags([-e,e],[0 1],n1-1,n1); % D block of [-1 1]
    D = [D;D(n1-1,:)];                 % Replicates last row to keep D sqr
    E = speye(n2);                     % E is n2 X n2 identity matrix
    Dy = kron(E,D);                    % Puts a D block at each diagonal
  else
    Dy = [];
  end;
    
else; % Compact shape

  % Generate Dx matrix
  if n1Dx > 0
    e = ones(n2,1);                    % e is an n2 vector of ones
    T = spdiags([-e e],[0 1],n2-1,n2); % T gives pattern of -I,I blocks in Dx
    E = speye(n1);                     % E is n1 X n1 identity matrix
    Dx = kron(T,E);                    % Puts an E block at each +-1 of T
  else
    Dx = [];
  end
 
  % Generate Dy matrix
  if n1Dy > 0
    e = ones(n1,1);                    % e is n1 vector of ones
    T = spdiags([-e,e],[0 1],n1-1,n1); % D block of [-1 1]
    E = speye(n2);
    Dy = kron(E,T);            % Puts a D block at each diagonal
  else
    Dy = [];
  end;

end
