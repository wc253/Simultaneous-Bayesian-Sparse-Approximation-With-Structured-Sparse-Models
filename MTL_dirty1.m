function [X_hat] = MTL_dirty1(CellPhi,Y,nu,alpha,max_iters)

% *************************************************************************
% Reweighted L2 algorithm for SSM-1
% CellPhi=cell(1,num_task);   
% Y = zeros(num_measurements, num_task);  
% nu is noise variance;
% alpha is the trade-off parameter for two sparse structures
% max_iters is the maximum number of iterations
% *************************************************************************


% *** Initializations ***
[N K] = size(Y);
[N M] = size(CellPhi{1,1});
mu_c = zeros(M,K);
z_c = ones(M,K);
z_a = ones(M,K);
gamma_a = ones(M,K);
gamma_c = ones(M,1);

% *** Control parameters ***
min_dmu = 1e-8;
current_itr = 0;


% *** Learning loop ***
while (1)
	mu_c_old = mu_c;
    
    gamma_c_rep = repmat(gamma_c,1,K);
    gamma_ac = (gamma_c_rep.*gamma_a)./(gamma_c_rep+gamma_a+ 1e-16);
    
    
    for i = 1:K

        % *** a innovations support Y covariance ***
        G_a = repmat(gamma_a(:,i)',N,1);
        PhiG_a = CellPhi{1,i}.*G_a;    
        Sigma_y_a = PhiG_a*CellPhi{1,i}' + (0.5*nu + 1e-12)*eye(N);
  
        
        % *** c support Y covariance *** 
        G_c = repmat(gamma_c',N,1);
        PhiG_c = CellPhi{1,i}.*G_c; 
        Sigma_y_c = PhiG_c*CellPhi{1,i}' + (0.5*nu + 1e-12)*eye(N);  
        
        % *** ac support Y covariance *** 
        G_ac = repmat(gamma_ac(:,i)',N,1);
        PhiG_ac = CellPhi{1,i}.*G_ac; 
        Sigma_y_ac = PhiG_ac*CellPhi{1,i}' + (nu + 1e-12)*eye(N);  
    
        Sigma_y = Sigma_y_ac;%
        Xi = real( Sigma_y\Y(:,i) );
        
        % *** Posterior means ***
        mu_c(:,i) = PhiG_ac'*Xi;

    
        % *** z_a ***
        Xi = real( Sigma_y_a\PhiG_a);
        z_a(:,i) = gamma_a(:,i) - sum(PhiG_a'.*Xi',2);        
        
        % *** z_c ***
        Xi = real( Sigma_y_c\PhiG_c);
        z_c(:,i) = gamma_c - sum(PhiG_c'.*Xi',2);
        
        % *** gamma_a ***
        gamma_a(:,i) = z_a(:,i)+(mu_c(:,i).^2)/alpha;        
    end;
    
    % *** gamma_c ***
    gamma_c = sum(z_c+mu_c.^2,2)/K;  
    
           
    % *** Check stopping conditions, etc. ***
  	current_itr = current_itr+1;   
    if (current_itr >= max_iters) break;  end; 
	dmu = max(max(abs( mu_c_old  - mu_c )));
    if (dmu < min_dmu)  break;  end;
end
X_hat = mu_c;
  
return;

