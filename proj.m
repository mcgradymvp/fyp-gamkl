function y=proj(x)

n=length(x);
s=sort(x,'descend');

flag=false;

for i=1:n-1
    m=(sum(s(1:i))-1)/i;
    if m>=s(i+1)
        flag=true;
        break;
    end
end

if flag==false
    m=(sum(s)-1)/n;
end

y=max(x-m,0);
end

    
