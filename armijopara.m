function [time, acc]=armijopara(dataset)
time = zeros(5, 5);
acc = zeros(5, 5);
avgtime = zeros(5, 5);
avgacc = zeros(5, 5);
rng('shuffle');
for i = 1:10
   iter = 1; 
   mat = -5.4999 + (5.4999*2)*rand(3,4);
   xt = [(1:4); mat];
   x = reshape(xt,[1,16]);
   for a = 0.3 : 0.1 : 0.7
       for b = 0.3 : 0.1 : 0.7
           m = cgd(dataset, x, [a, b]);
           time(iter) = m.time;
           acc(iter) = m.acc;
           iter = iter + 1;
       end
   end
   avgtime = avgtime + time;
   avgacc = avgacc + acc;
end

time = avgtime / 10;
acc = avgacc / 10;
