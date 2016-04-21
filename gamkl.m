function [x, fval, exitflag] = gamkl(dataset, num_ker)
%This function is used to run the GA based MKL algorithm. The algorithm
%aims to find a set of best suitable set of kernels, which will return a
%desent classification performance.
% [x, fval, exitflag] = gamkl(dataset, num_ker)
% Input
% dataset: A structure type of data set.
% num_ker: The number of kernels

startTime = tic;

fun=@(para) -fitness(dataset,para);
opts = gaoptimset('PlotFcns',@gaplotbestf,'PopulationSize',40,...
    'Generations',100,'StallGenLimit',30,...
    'UseParallel',true,'Display','iter');
lb=repmat([1;-5.4999;-5.4999;-5.4999],[num_ker,1]);
ub=repmat([4;5.4999;5.4999;5.4999],[num_ker,1]);
ker_lable=1:4:4*num_ker;

if max(size(gcp)) == 0 
    parpool
end

[x, fval, exitflag]=ga(fun,4*num_ker,[],[],[],[],lb,ub,[],ker_lable,opts);

time_ga_parallel = toc(startTime);
fprintf('Parallel GA optimization takes %g seconds.\n',time_ga_parallel);

end

function result=fitness(data,opti)
model=pgd(data,opti);
result=model.acc;
end
