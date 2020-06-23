clc,clear all
%% MISO YSA 2017
load data

% YSA Model
[N,R] = size(T);            % N =4992 toplam data sayisi R = 9 Toplam Giris Sayisi 9 veri boyutu
S = 50;                      % Nöron Sayisi  

Wg = rand(S,R)-0.5;         % Giris agirlik matrisi baslangiçlari random -0.5 +0.5 araligi
bh = rand(S,1)-0.5;         % Hidden Layer Biaslar
Wc = rand(1,S)-0.5;         % Çikis agirlik matrisi
bc = rand(1,1)-0.5;         % Output bias

%% Datalari Training ve Validation olarak 2 ye bölüyoruz.

TrainingIndex   = 1:2:N;
ValidationIndex = 2:2:N;
TrainingINPUT    = T(TrainingIndex,:);
TrainingOUTPUT   = Y(TrainingIndex,:);
ValidationINPUT  = T(ValidationIndex,:);
ValidationOUTPUT = Y(ValidationIndex,:);
Ntraining   = size(TrainingINPUT,1);
Nvalidation = size(ValidationINPUT,1);

if S*(R+2)+1 > Ntraining    % Ysa Parametre Sayisi Parametre sayisi Data sayisini geçemez!
    disp('too much neurons!');
    return
end

Nmax = 20;
I = eye(S*(R+2)+1);
kosul = 1; iteration = 0; mu = 1; FvalMIN = inf;
while kosul
    iteration = iteration + 1;
    [ yhat ] = MISOYSAmodel( TrainingINPUT,Wg,bh,Wc,bc);
    eTra = TrainingOUTPUT - yhat;                               
    f = eTra'*eTra;
    J = [];
    for i=1:Ntraining
        for j=S*(R+2)+1
            J(i,j) = -1;
        end
        for j=S*(R+1)+1:S*(R+2)
            J(i,j) = -tanh(Wg(j-(R+1)*S,:)*TrainingINPUT(i,:)'+bh(j-(R+1)*S));
        end
        for j=S*R+1:S*R+S
            J(i,j) = -Wc(1,j-S*R)*(1)*[1-tanh(Wg(j-S*R,:)*TrainingINPUT(i,:)'+bh(j-S*R))^2];
        end
        for j=1:S*R
            k = mod(j-1,S)+1;
            m = fix((j-1)/S)+1;
            J(i,j) = -Wc(1,k)*TrainingINPUT(i,m)*[1-tanh(Wg(k,:)*TrainingINPUT(i,:)'+bh(k))^2];
        end
    end
    loop2 = 1;
    while loop2
        p = -inv(J'*J+mu*I)*J'*eTra;
        [x] = matrix2vector( Wg,bh,Wc,bc );
        [Wgz,bhz,Wcz,bcz] = vector2matrix( x+p,S,R );
        [yhatz] = MISOYSAmodel( TrainingINPUT,Wgz,bhz,Wcz,bcz);
        fz = (TrainingOUTPUT-yhatz)'*(TrainingOUTPUT-yhatz);
        if fz <f
            x = x +p;
            [Wg,bh,Wc,bc] = vector2matrix( x,S,R );
            mu = 0.1*mu;
            loop2 = 0;
        else
            mu = 10*mu;
            if mu>1e+20
                loop2 = 0;
                kosul = 0;
            end
        end
    end
    
    [ yhat ] = MISOYSAmodel( TrainingINPUT,Wg,bh,Wc,bc);
    eTra = TrainingOUTPUT - yhat;
    f = eTra'*eTra;
    FTRAINING(iteration) = f;
    
    [yhat] = MISOYSAmodel( ValidationINPUT,Wg,bh,Wc,bc);
    eVal = ValidationOUTPUT - yhat;
    fVALIDATION = eVal'*eVal;
    FVALIDATION(iteration) = fVALIDATION;
    if fVALIDATION < FvalMIN
        xbest = x;
        FvalMIN = fVALIDATION;
    end
    g = 2*J'*eTra;
    fprintf('iteration:%4.0f\t ||g||:%4.6f\t ftr:%4.6f\t fv:%4.6f\n',([iteration norm(g) f fVALIDATION]));
    if[iteration>=Nmax]
        kosul = 0;
    end
end
%% PLOT DATA
[Wg,bh,Wc,bc] = vector2matrix(x,S,R);
[yhatTR] = MISOYSAmodel(TrainingINPUT,Wg,bh,Wc,bc);
[Wg,bh,Wc,bc] = vector2matrix(xbest,S,R);
[yhatVA] = MISOYSAmodel(ValidationINPUT,Wg,bh,Wc,bc);

figure
subplot(221)
plot(TrainingOUTPUT,'r','LineWidth',2);
hold on,grid minor,title('Neural Network Training Output ')
plot(yhatTR,'k--','LineWidth',2);
legend('=Training Data','=Training Output')


subplot(222)
plot(ValidationOUTPUT,'r','LineWidth',2);
hold on,grid minor,title('Neural Network Validation Output ')
plot(yhatVA,'k--','LineWidth',2);
legend('=Validation Data','=Validation Output')


subplot(223)
plot(FTRAINING,'b')
hold on,grid minor
plot(FVALIDATION,'r')
legend('=FTRAINING','=FVALIDATION')
title(['Number Of Neuron:', num2str(S), 'Fvalbest=', num2str(FvalMIN)])
 

subplot(224)
ValErr = ValidationOUTPUT - yhatVA;
plot(ValErr,'r','LineWidth',2), grid minor
title(['Output Error',' RMS ',num2str(rms(eVal))])



