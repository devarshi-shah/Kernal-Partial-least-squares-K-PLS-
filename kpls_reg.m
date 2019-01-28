%% this is KPLS algorithm used when Y is a matrix.
%%% output: B = regression coefficients, Tf= scores of kernel PLS, Uf= projection vector, Xfc= centrelized
%%% kerenel required for future prediction, Yf=original y after scaling, myf= mean of y, syf= standar deviation of y 


function [B,Tf,Uf,Xfc,Yf,myf,syf]=kpls_reg(Xf1,yf1,npcf)



rankx=rank(Xf1);
if rankx<npcf
    disp(sprintf('Rank of X is %g, which is less than input LVs %g.',rankx,npcf))
    npcf=rankx;
end
[nsp,nid]=size(Xf1);
[~,nd]=size(yf1);
Tf=zeros(nsp,npcf);

I=eye(nsp);
O=ones(nsp);
[yf1,myf,syf]=zscore(yf1);
Yf=yf1;
Xf1=(I-((1/nid)*O))*Xf1*(I-((1/nid)*O));
Xfc=Xf1;
I=eye(nsp);

for j=1:npcf
    
     if nd > 1
         ssy = sum(yf1.^2);
         [~,yi] = max(ssy);
         u1 = yf1(:,yi);
     else
         u1=yf1(:,1);
     end
rng(5);
unew=rand(nsp,1);
diff=u1-unew;
 while ((norm(diff))/(norm(unew))>10^-10)
     
    t=Xf1*u1;
    t=t/norm(t);
    
    c=yf1'*t;
    u=yf1*c;
    
    u=u/norm(u);
    unew=u;
    diff=u1-unew;
    u1=u;   
    
 end
Tf(:,j)=t;
Uf(:,j)=u;

Xf1=(I-t*t')*Xf1*(I-t*t');
yf1=yf1-(t*t'*yf1);

B(:,j)=Uf(:,1:j)*inv(Tf(:,1:j)'*Xfc*Uf(:,1:j))*(Tf(:,1:j))'*Yf;
end
 
%%% use Tf, Uf, Xfc (kernel matrix) & Yf for prediction of both test and training sets. check paper Kernel Partial Least Squares Regression in Reproducing
%%% Kernel Hilbert Space by Rosipal
