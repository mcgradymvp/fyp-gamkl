load('heart.mat');
heart=heart_scale;
xtrain=heart.x;
ytrain=heart.y;

NIND = 50;
MAXGEN = 100;
GGAP = 0.9;
NVAR = 3;
PRECI = 10;

FieldD = [rep([PRECI],[1, NVAR]); rep([-10;10],[1, NVAR]);...
    rep([1; 0; 1 ;1], [1, NVAR])];

Chrom = crtbp(NIND, NVAR*PRECI);

Best = NaN*ones(MAXGEN,1);
gen = 0;

temp=bs2rv(Chrom,FieldD);
ObjV=zeros(NIND,1);
for i=1:NIND 
    ObjV(i)=fitness(heart,kernels,temp(i,:));
end

Best(gen+1) = max(ObjV);
plot(Best,'ro');xlabel('generation'); ylabel('f(x)');
text(0.5,0.95,['Best = ', num2str(Best(gen+1))],'Units','normalized');
drawnow; 

while gen < MAXGEN
    
    FitnV = ranking(-ObjV);
    SelCh = select('sus', Chrom, FitnV, GGAP);
    SelCh = recombin('xovsp', SelCh, 0.7);
    SelCh = mut(SelCh);
    
    [m, ~]=size(SelCh);
    temp=bs2rv(SelCh,FieldD);
    ObjVSel=zeros(m,1);
    for i=1:m
        ObjVSel(i)=fitness(heart,kernels,temp(i,:));
    end
    
    [Chrom, ObjV]=reins(Chrom, SelCh, 1, 1, ObjV, ObjVSel);
    
    gen = gen+1;
    
    Best(gen+1) = max(ObjV);
    
    plot(Best,'ro'); xlabel('generation'); ylabel('f(x)');
    text(0.5,0.95,['Best = ', num2str(Best(gen+1))],'Units','normalized');
    drawnow;
    %if (Best(gen+1)-Best(gen))<10e-6
    %    break
    %end
end
   

         