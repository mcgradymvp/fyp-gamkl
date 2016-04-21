function H=myker(X,Y,kernels,para)
%This function is used to compute kernel matrix. 
% H = myker(X,Y,kernels,para)
% Input
% kernels: label for the kernel type
% para:    three parameters which will be mapped according to 
%          the type of kernel

switch kernels
    case {1}
        %Polynomial kernel
        a=para(1);
        b=para(2);
        d=round(para(3));
        if isempty(Y)
            H = (a*(X * X') + b).^d;
        else
            H = (a*(X * Y') + b).^d;
        end
        
    case {2}
        %Radial kernel
        sigma=mean(para);
        if isempty(Y)
            sx = sum(X'.^2, 1);
            D = bsxfun(@minus, bsxfun(@plus, sx.', sx), 2 * (X * X'));
        else
            sx = sum(X'.^2, 1);
            sy = sum(Y'.^2, 1);
            D = bsxfun(@minus, bsxfun(@plus, sx.', sy), 2 * (X * Y'));
        end
        
        H = exp(- D * (1 / (2 * sigma^2)));
        
    case {3}
        %T-Student kernel
        degree=abs(mean(para));
        if isempty(Y)
            sx = sum(X'.^2, 1);
            D = sqrt(bsxfun(@minus, bsxfun(@plus, sx.', sx),...
                2 * (X * X')));
        else
            sx = sum(X'.^2, 1);
            sy = sum(Y'.^2, 1);
            D = sqrt(bsxfun(@minus, bsxfun(@plus, sx.', sy),...
                2 * (X * Y')));
        end
        
        H = 1./(1+D.^degree);
        
    case {4}
        %Wavelet kernel
        a=mean(para);
        n=size(X,1);
        if isempty(Y),
            XXh = sum(X.^2,2)*ones(1,n);
            omega = XXh+XXh'-2*(X*X');
            
            XXh1 = sum(X,2)*ones(1,n);
            omega1 = XXh1-XXh1';
            H = cos(1.75*omega1./a).*exp(-omega./(2*a^2));
            
        else
            XXh1 = sum(X.^2,2)*ones(1,size(Y,1));
            XXh2 = sum(Y.^2,2)*ones(1,n);
            omega = XXh1+XXh2' - 2*(X*Y');
            
            XXh11 = sum(X,2)*ones(1,size(Y,1));
            XXh22 = sum(Y,2)*ones(1,n);
            omega1 = XXh11-XXh22';
            
            H = cos(1.75*omega1./a).*exp(-omega./(2*a^2));
        end
end

end