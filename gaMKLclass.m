function [BestCVaccuracy, Bestx, gaoption] = gaMKLclass(dataset, num_ker, gaoption)
tic;

if nargin == 2
    gaoption = struct('maxgen',100,'sizepop',40,'ggap',0.9,...
        'bound',[0.5,-5.4999,-5.4999,-5.4999;4.4999,5.4999,5.4999,5.4999]);
end


MAXGEN = gaoption.maxgen;
NIND = gaoption.sizepop;
NVAR = 4 * num_ker;
PRECI = 20;
GGAP = gaoption.ggap;
trace = zeros(MAXGEN, 2);
bound = gaoption.bound;

FieldID = ...
    [rep([PRECI],[1,NVAR]);...
    rep(bound,[1,num_ker]);rep([1;0;1;1],[1,NVAR])];

Chrom = crtbp(NIND,NVAR*PRECI);

gen = 1;
BestCVaccuracy = 0;
Bestx = 0;

tempx = bs2rv(Chrom,FieldID);
for i = 1:num_ker
    tempx(:,4*i-3) = round(tempx(:,4*i-3));
end
 
if max(size(gcp)) == 0 % parallel pool needed
    parpool % create the parallel pool
end

parfor nind = 1:NIND
    model = pgd(dataset, tempx(nind,:));
    ObjV(nind,1) = model.acc;
end

[BestCVaccuracy,I] = max(ObjV);
Bestx = tempx(I,:);

for  gen = 1:MAXGEN
    FitnV = ranking(-ObjV);
    
    SelCh = select('sus',Chrom,FitnV,GGAP);
    SelCh = recombin('xovsp',SelCh,0.7);
    SelCh = mut(SelCh);
    
    tempx = bs2rv(SelCh,FieldID);
    for i = 1:num_ker
        tempx(:,4*i-3) = round(tempx(:,4*i-3));
    end
    
    parfor nind = 1:size(SelCh,1)
        model = pgd(dataset, tempx(nind,:));
        ObjVSel(nind,1) = model.acc;
    end
    
    [Chrom,ObjV] = reins(Chrom,SelCh,1,1,ObjV,ObjVSel);
    
    if max(ObjV) <= 50
        continue;
    end
    
    [NewBestCVaccuracy,I] = max(ObjV);
    x_temp = bs2rv(Chrom,FieldID);
    temp_NewBestCVaccuracy = NewBestCVaccuracy;
    
    if NewBestCVaccuracy > BestCVaccuracy
       BestCVaccuracy = NewBestCVaccuracy;
       Bestx = x_temp(I,:);
    end 
    
    if abs( NewBestCVaccuracy-BestCVaccuracy ) <= 10^(-2)
       BestCVaccuracy = NewBestCVaccuracy;
       Bestx = x_temp(I,:);
    end 
    
    trace(gen,1) = max(ObjV);
    trace(gen,2) = sum(ObjV)/length(ObjV);
    
    gen = gen+1;
    
    if gen <= MAXGEN/2
        continue;
    end
    if BestCVaccuracy >=80 && ...
       ( temp_NewBestCVaccuracy-BestCVaccuracy ) <= 10^(-2)     
        break;
    end
    if gen == MAXGEN
        break;
    end
end
gen = gen - 1;

figure;
hold on;
trace = round(trace*10000)/10000;
plot(trace(1:gen,1),'r*-','LineWidth',1.5);
plot(trace(1:gen,2),'o-','LineWidth',1.5);
legend('Best Fitness','Average Fitness',3);
xlabel('Generation','FontSize',12);
ylabel('Fitness Value','FontSize',12);
axis([0 gen 0 100]);
grid on;
axis auto;

line1 = 'Fitness Curve';
title({line1},'FontSize',12);
toc;