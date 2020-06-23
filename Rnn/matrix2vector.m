function [ x ] = matrix2vector( Wg,bh,Wc,bc )
R = size(Wg,2);
x = [];  
for r=1:R
    x = [x; Wg(:,r) ];
end
x = [x; bh];
x = [x; Wc'];
x = [x; bc];
end

