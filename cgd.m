function model = cgd(data, opti, armijopara, maxiter)

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
    gram_matrix = gram_matrix + mu(i) * myker(data.x, [], kernels(i), para(i,:));
end

stepsize = 1;

for iter = 2 : maxiter+1

    model = svmtrain2(data.y, [(1:n)' gram_matrix], '-t 4 -q');
    J(iter) = -model.obj;
    ay = zeros(n, 1);
    ay(model.sv_indices) = model.sv_coef;
        
    for i = 1 : M
        dJ(i, iter) = -0.5 * ay' * myker(data.x, data.x, kernels(i), para(i,:)) * ay;
    end
 
%     disp(iter);
%     disp(J(iter));
%     disp(dJ(:, iter));
    
    mubar = linprog(dJ(:, iter), ones(1, M), 1, [], [], zeros(M, 1),...
        [],[],optimset('Display','off'));
    
    if abs(dJ(:, iter)' * (mubar - mu)) <= 10e-8...
            || abs(J(iter) - J(iter-1)) <= 10e-8;
        break;
    end
    
    d = mubar - mu;
    armijo_sigma = armijopara(1);
    armijo_beta = armijopara(2);
    
    while true
        mu_hat = mu + stepsize * d;
        gram_matrix = zeros(n, n);
        for i = 1 : M
            gram_matrix = gram_matrix +...
                mu_hat(i) * myker(data.x, data.x, kernels(i), para(i,:));
        end
        
        model = svmtrain2(data.y, [(1:n)' gram_matrix], '-t 4 -q');
        J(iter + 1) = -model.obj;
        lhs = J(iter+1);
        rhs = J(iter) + armijo_sigma * stepsize * dJ(:, iter)' * d;
        
        if lhs <= rhs
            %stepsize = min(1,stepsize/armijo_beta);
            stepsize = 1;
            break;
        end
        
        stepsize = stepsize * armijo_beta;
    end
    mu = mu_hat;
end

H=zeros(n,n);
for i = 1:M
    H = H + mu(i) * myker(data.x, data.x, kernels(i), para(i,:));
end

accuracy=svmtrain(data.y,[(1:n)' H],'-t 4 -v 5 -q');

model.mu=mu;

if flag == true
    model.acc=0;
else
    model.acc=accuracy;
end

model.iteration=iter-1;

time=toc;
model.time=time;
end