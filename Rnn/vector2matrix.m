function [ Wg,bh,Wc,bc ] = vector2matrix( x,S,R )
Wg = []; 
for r=1:R
    Wg = [Wg, x((r-1)*S+1:r*S) ];
end
bh(:,1) = x(S*R+1:S*R+S);
Wc(1,:) = x(S*(R+1)+1:S*(R+2));
bc = x(S*(R+2)+1);