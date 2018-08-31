function [mu_c,mu_s] = MTL_dirty2(CellPhi,Y,nu,alpha,max_iters)

% *************************************************************************
% Reweighted L2 algorithm for the SSM-2
% CellPhi=cell(1,num_task);   
% Y = zeros(num_measurements, num_task);  
% nu is noise variance;
% alpha is the trade-off parameter for two sparse structures
% max_iters is the maximum number of iterations
% mu_c is the row sparse component
% mu_s is the element sparse component
% *************************************************************************


% *** Initializations ***
[N K] = size(Y);
[N M] = size(CellPhi{1,1});
mu_c = zeros(M,K);
mu_s = zeros(M,K);
gamma_s = ones(M,K);
z_c = ones(M,K);
z_s = ones(M,K);

% *** Control parameters ***
min_dmu = 1e-8;
current_itr = 0;

% *** Warm start ***
for i=1:K
    [mu,dmu,k,gamma] = sparse_learning(CellPhi{1,i},Y(:,i),nu,5,2,0,0);
    mu_s(:,i) = 0.5*mu;
    mu_c(:,i) = 0.5*mu;
    gamma_s(:,i) = 0.5*gamma;
end
gamma_c = sum(gamma_s,2);

% *** Learning loop ***
while (1)
	mu_c_old = mu_c;
    mu_s_old = mu_s;
    
    for i = 1:K
        % *** Common support Y covariance ***
        G_c = repmat(gamma_c',N,1);
        PhiG_c = CellPhi{1,i}.*G_c; 
        Sigma_y_c = PhiG_c*CellPhi{1,i}' + (0.5*nu + 1e-12)*eye(N);
    
        % *** Sparse innovations support Y covariance ***
        G_s = repmat(gamma_s(:,i)',N,1);
        PhiG_s = CellPhi{1,i}.*G_s;    
        Sigma_y_s = PhiG_s*CellPhi{1,i}' + (0.5*nu + 1e-12)*eye(N);
    
        Sigma_y = Sigma_y_c + Sigma_y_s;
        Xi = real( Sigma_y\Y(:,i) );
        
        % *** Posterior means ***
        mu_c(:,i) = PhiG_c'*Xi;
        mu_s(:,i) = PhiG_s'*Xi;
    
        % *** z_s ***
        Xi = real( Sigma_y_s\PhiG_s);
        z_s(:,i) = gamma_s(:,i) - sum(PhiG_s'.*Xi',2);
        
        % *** z_c ***
        Xi = real( Sigma_y_c\PhiG_c);
        z_c(:,i) = alpha*(gamma_c - sum(PhiG_c'.*Xi',2));
        
        % *** gamma_s ***
        gamma_s(:,i) = z_s(:,i)+mu_s(:,i).^2;
    end;
    
    % *** gamma_c ***
    gamma_c = sum(z_c+abs(mu_c).^2,2)/K/alpha;  
    
           
    % *** Check stopping conditions, etc. ***
  	current_itr = current_itr+1;   
    if (current_itr >= max_iters) break;  end; 
	dmu = max(max(abs( mu_c_old + mu_s_old - mu_c - mu_s )));
    if (dmu < min_dmu)  break;  end;
end

  
return;

