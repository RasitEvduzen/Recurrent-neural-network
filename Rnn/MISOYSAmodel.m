function [ yhat ] = MISOYSAmodel( T,Wg,bh,Wc,bc )
[N,R] = size(T) ;
for n=1:N
    yhat(n,1) = Wc * tanh(Wg*T(n,:)'+bh)+bc;
end
end

