function [avacc, avtime] = testarmijo(dataset, x)
acc = zeros(1, 10);
time = zeros(1, 10);

parfor i = 1: 10
    % m = pgd(dataset, x(i, :));
    m = cgd(dataset, x(i, :));
    acc(i) = m.acc;
    time(i) = m.time;
end

avacc = mean(acc);
avtime = mean(time);
end
