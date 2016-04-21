function model = pgd(data, opti, armijopara, maxiter)
%This function is used to calculate weight of each kernel using gradient
%projection method.
% model = pgd(data, opti, armijopara, maxiter)
% Input
% data: data set with structure type
%       data.x is the features and data.y is the labels
% opti: parameters for different kernels
%       for each kernel there are four parameters, the first is the kernel
%       label and the remaining three will be mapped according to the type
%       of kernel
% armijopara: choice of Armijo parameters, by default sigma=beta=0.4
% maxiter: maximum iteration of the gradient projection method
%%
tic;
M = length(opti)/4;
kernels = opti(1:4:4*M);
para = zeros(M, 3);
for i = 1:M
    para(i, :)=opti(4 * i - 2 : 4 * i);
end

if nargin < 4
    maxiter = 200;
    if nargin < 3
        armijopara = [0.4, 0.4];
    end
end

flag = false;

[n, ~] = size(data.x);
dJ = zeros(M, maxiter);
J = zeros(1, maxiter);

mu = 1/M * ones(M, 1);
gram_matrix = zeros(n, n);
for i = 1:M
    gram_matrix = gram_matrix + mu(i) * myker(data.x, [], kernels(i),...
        para(i,:));
end
armijo_s = 1;
count=zeros(maxiter,1);

for iter = 2 : maxiter + 1
    
    count_m=0;
    model = svmtrain2(data.y, [(1:n)' gram_matrix], '-t 4 -q');
    J(iter) = -model.obj;
    ay = zeros(n, 1);
    ay(model.sv_indices) = model.sv_coef;
   
    for i = 1 : M
        dJ(i, iter) = -0.5 * ay' * myker(data.x, data.x, kernels(i),...
            para(i,:)) * ay;
    end
    
    if abs(J(iter) - J(iter-1)) <= 10e-8;    %Stopping criterion
        break;
    end
    %% Armijo Rule to find the step size
    armijo_sigma = armijopara(1);
    armijo_beta = armijopara(2);
    while true
        if count_m >= 20
            flag = true;
            break;
        end
        
        mu_hat = proj(mu-armijo_s*dJ(:,iter));
        gram_matrix = zeros(n, n);
        for i = 1 : M
            gram_matrix = gram_matrix +...
                mu_hat(i) * myker(data.x, data.x, kernels(i), para(i,:));
        end
        
        model = svmtrain2(data.y, [(1:n)' gram_matrix], '-t 4 -q ');
        J(iter + 1) = -model.obj;
        lhs = J(iter+1) + armijo_sigma * dJ(:,iter)' * (mu-mu_hat);
        rhs = J(iter);
        
        if lhs <= rhs
            armijo_s = min(1,armijo_s/armijo_beta); %modified Armijo Rule
            % armijo_s = 1;                         %Original Armijo Rule
            break;
        end
        
        armijo_s = armijo_s * armijo_beta;
        count_m = count_m + 1;
    end
    %%
    mu = mu_hat;
    count(iter-1) = count_m;   
    if flag == true
        break;
    end
end

H=zeros(n,n);
for i = 1:M
    H = H + mu(i) * myker(data.x, data.x, kernels(i), para(i,:));
end

accuracy=svmtrain(data.y,[(1:n)' H],'-t 4 -v 5 -q ');

model.mu=mu;
model.acc=accuracy;
model.iteration=iter-1;
time=toc;
model.time=time;
model.count=count;

end