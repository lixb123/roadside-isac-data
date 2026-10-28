function [sol,info]=solve_inherited(ch,cfg,init)
% Refactored user-supplied AO; Li et al. TWC2024 DOI3376976.
% W is normalized: sum(abs(W).^2)<=1; physical beams=sqrt(powerW)*W.
boundChannel=norm(ch.hs(:,4))+sum(abs(ch.hr(4,:)).*vecnorm(ch.Hsr,2,2)');
sensingNoise=cfg.noiseW;if isfield(cfg,'sensingNoiseW'),sensingNoise=cfg.sensingNoiseW;end
echoUpper=cfg.rcs*cfg.powerW/sensingNoise*boundChannel^4;
if isfinite(echoUpper) && echoUpper<cfg.echoThreshold
 sol=init;m=evaluate_isac(ch,sol,cfg);
 info=struct('runtime',0,'status',{{'InfeasibleByEchoBound'}},'failed',true, ...
  'errorMessage',"Echo upper bound below threshold",'metrics',m,'accepted',false, ...
  'repaired',false,'returnedMetrics',m,'echoUpperBound',echoUpper);
 return;
end
M=cfg.M; N=cfg.N; P=1; sigma=1; scale=sqrt(cfg.powerW/cfg.noiseW);
echoFactor=cfg.rcs/scale^2*cfg.noiseW/sensingNoise;
Hsr=ch.Hsr; hs1=ch.hs(:,1); hs2=ch.hs(:,2); hs3=ch.hs(:,3); hst=ch.hs(:,4);
hr1=scale*ch.hr(1,:); hr2=scale*ch.hr(2,:); hr3=scale*ch.hr(3,:); hrt=scale*ch.hr(4,:); hs1=scale*hs1;hs2=scale*hs2;hs3=scale*hs3;hst=scale*hst;
v10=init.W(:,1);v20=init.W(:,2);v30=init.W(:,3);vj0=init.W(:,4);V0=init.V;
v1=v10;v2=v20;v3=v30;vj=vj0;V=V0;
r1=cfg.rateMin;r2=r1;r3=r1;
jamMin=1;if isfield(cfg,'jamRateMin'),jamMin=cfg.jamRateMin;end
t10=2;t20=2;t30=2;e10=.99;e20=.99;e30=.99;
t_20=2;t_30=2;t_40=2;t_50=2;sa10=2;sa20=2;sa30=2;
failed=false;logStatus={}; timer=tic; last=init; repaired=false; errorMessage="";
try
for k=1:cfg.outer


h1 = [V0' 1]*[diag(hr1)*Hsr ; hs1'];
h2 = [V0' 1]*[diag(hr2)*Hsr ; hs2'];
h3 = [V0' 1]*[diag(hr3)*Hsr ; hs3'];
he = [V0' 1]*[diag(hrt)*Hsr ; hst'];
if isfield(cfg,'initializeAux') && cfg.initializeAux
 Z=abs([h1;h2;h3;he]*[v10 v20 v30 vj0]).^2;
 tv=zeros(1,3);ev=zeros(1,3);
 for stream=1:3
  tv(stream)=1+min(Z(stream:3,stream)./(sum(Z(stream:3,stream+1:3),2)+sigma));
  ev(stream)=(sum(Z(4,:))-Z(4,stream)+sigma)/(sum(Z(4,:))+sigma);
 end
 t10=max(tv(1),1.001);t20=max(tv(2),1.001);t30=max(tv(3),1.001);
 e10=ev(1);e20=ev(2);e30=ev(3);
 t_20=sqrt(t10*t20);t_30=sqrt(t30*t_20^2);
 t_40=sqrt(e10*t_30^2);t_50=sqrt(e20*t_40^2);
 sav=1+Z(1:3,4)./(sum(Z(1:3,1:3),2)+sigma);
 sa10=max(sav(1),1.001);sa20=max(sav(2),1.001);sa30=max(sav(3),1.001);
end






for i=1:cfg.inner
cvx_begin quiet
cvx_solver Mosek

variable v1(M,1) complex
variable v2(M,1) complex
variable v3(M,1) complex
variable vj(M,1) complex
variables e1 e2 e3 t1 t2 t3 t_6 Vj
variables t_2 t_3 t_4 t_5 z_2 z_3 z_4 z_5
variables sa1 sa2 sa3
maximize t_6
subject to
t_6>=0;

t1>= 2^r1;
t2>= 2^r2;
t3>= 2^r3;

norm([2*t_6,(e3-z_5)]')<=e3+z_5;
z_5<=t_50^2+2*t_50*(t_5-t_50);
norm([2*t_5,(e2-z_4)]')<=e2+z_4;
z_4<=t_40^2+2*t_40*(t_4-t_40);
norm([2*t_4,(e1-z_3)]')<=e1+z_3;
z_3<=t_30^2+2*t_30*(t_3-t_30);
norm([2*t_2,(t1-t2)]')<=t1+t2;
norm([2*t_3,(t3-z_2)]')<=t3+z_2;
z_2<=t_20^2+2*t_20*(t_2-t_20);

g11=2*real(v10'*h1'*h1*v1)/(t10-1)-real(v10'*h1'*h1*v10)*(t1-1)/(t10-1)^2-sigma;
g21=2*real(v10'*h2'*h2*v1)/(t10-1)-real(v10'*h2'*h2*v10)*(t1-1)/(t10-1)^2-sigma;
g31=2*real(v10'*h3'*h3*v1)/(t10-1)-real(v10'*h3'*h3*v10)*(t1-1)/(t10-1)^2-sigma;
norm([2*h1*v2,2*h1*v3,(g11-1)]') <= g11+1;
norm([2*h2*v2,2*h2*v3,(g21-1)]') <= g21+1;
norm([2*h3*v2,2*h3*v3,(g31-1)]') <= g31+1;

g22=2*real(v20'*h2'*h2*v2)/(t20-1)-real(v20'*h2'*h2*v20)*(t2-1)/(t20-1)^2-sigma;
g32=2*real(v20'*h3'*h3*v2)/(t20-1)-real(v20'*h3'*h3*v20)*(t2-1)/(t20-1)^2-sigma;
norm([2*h2*v3,(g22-1)]')<= g22+1;
norm([2*h3*v3,(g32-1)]')<= g32+1;

g33=2*real(v30'*h3'*h3*v3)/(t30-1)-real(v30'*h3'*h3*v30)*(t3-1)/(t30-1)^2-sigma;
0<=g33;

q21=2*real(v20'*he'*he*v2)/(e10)-real(v20'*he'*he*v20)*(e1)/(e10)^2;
q31=2*real(v30'*he'*he*v3)/(e10)-real(v30'*he'*he*v30)*(e1)/(e10)^2;
qa1=2*real(vj0'*he'*he*vj)/(e10)-real(vj0'*he'*he*vj0)*(e1)/(e10)^2;
Q1=q21+q31+qa1  + 2*sigma/(e10)-sigma*(e1)/(e10)^2 - sigma;
norm([2*he*v1,2*he*v2,2*he*v3,2*he*vj,(Q1-1)]')<= Q1+1;

q12=2*real(v10'*he'*he*v1)/(e20)-real(v10'*he'*he*v10)*(e2)/(e20)^2;
q32=2*real(v30'*he'*he*v3)/(e20)-real(v30'*he'*he*v30)*(e2)/(e20)^2;
qa2=2*real(vj0'*he'*he*vj)/(e20)-real(vj0'*he'*he*vj0)*(e2)/(e20)^2;
Q2=q12+q32+qa2  + 2*sigma/(e20)-sigma*(e2)/(e20)^2 - sigma;
norm([2*he*v1,2*he*v2,2*he*v3,2*he*vj,(Q2-1)]')<= Q2+1;

q13=2*real(v10'*he'*he*v1)/(e30)-real(v10'*he'*he*v10)*(e3)/(e30)^2;
q23=2*real(v20'*he'*he*v2)/(e30)-real(v20'*he'*he*v20)*(e3)/(e30)^2;
qa3=2*real(vj0'*he'*he*vj)/(e30)-real(vj0'*he'*he*vj0)*(e3)/(e30)^2;
Q3=q13+q23+qa3  + 2*sigma/(e30)-sigma*(e3)/(e30)^2 - sigma;
norm([2*he*v1,2*he*v2,2*he*v3,2*he*vj,(Q3-1)]')<= Q3+1;


s1j=2*real(vj0'*h1'*h1*vj)-real(vj0'*h1'*h1*vj0);
s2j=2*real(vj0'*h2'*h2*vj)-real(vj0'*h2'*h2*vj0);
s3j=2*real(vj0'*h3'*h3*vj)-real(vj0'*h3'*h3*vj0);
s11=2*real(v10'*h1'*h1*v1)-real(v10'*h1'*h1*v10);
s21=2*real(v10'*h2'*h2*v1)-real(v10'*h2'*h2*v10);
s31=2*real(v10'*h3'*h3*v1)-real(v10'*h3'*h3*v10);
s22=2*real(v20'*h2'*h2*v2)-real(v20'*h2'*h2*v20);
s32=2*real(v20'*h3'*h3*v2)-real(v20'*h3'*h3*v20);
norm(h1*v1)<=sqrt(s1j);
norm(h1*v2)<=sqrt(s11);
norm(h1*v3)<=sqrt(s11);
norm(h2*v1)<=sqrt(s2j);
norm(h2*v2)<=sqrt(s21);
norm(h2*v3)<=sqrt(s22);
norm(h3*v1)<=sqrt(s3j);
norm(h3*v2)<=sqrt(s31);
norm(h3*v3)<=sqrt(s32);


Vj<=2*real(vj'*vj0)-real(vj0'*vj0);
Vj>=0;
max(norm(v1),norm(v2))<=sqrt(Vj);
max(norm(v2),norm(v3))<=sqrt(Vj);

norm([v1',v2',v3',vj']')<=sqrt(P);


   La1=(2*real(vj0'*h1'*h1*vj)/(sa10-1)-real(vj0'*h1'*h1*vj0)*(sa1-1)/(sa10-1)^2)-sigma;
   La2=(2*real(vj0'*h2'*h2*vj)/(sa20-1)-real(vj0'*h2'*h2*vj0)*(sa2-1)/(sa20-1)^2)-sigma;
   La3=(2*real(vj0'*h3'*h3*vj)/(sa30-1)-real(vj0'*h3'*h3*vj0)*(sa3-1)/(sa30-1)^2)-sigma;

   norm([2*(h1*v1),2*(h1*v2),2*(h1*v3),(La1-1)]')<=La1+1;
   norm([2*(h2*v1),2*(h2*v2),2*(h2*v3),(La2-1)]')<=La2+1;
   norm([2*(h3*v1),2*(h3*v2),2*(h3*v3),(La3-1)]')<=La3+1;

                sa1>=2^jamMin;sa2>=2^jamMin;sa3>=2^jamMin;


h1e=2*real(v10'*(he'*he)*(he*he')*v1)-real(v10'*(he'*he)*(he*he')*v10);
h2e=2*real(v20'*(he'*he)*(he*he')*v2)-real(v20'*(he'*he)*(he*he')*v20);
h3e=2*real(v30'*(he'*he)*(he*he')*v3)-real(v30'*(he'*he)*(he*he')*v30);
hae=2*real(vj0'*(he'*he)*(he*he')*vj)-real(vj0'*(he'*he)*(he*he')*vj0);
echoFactor*(h1e+h2e+h3e+hae) >= cfg.echoThreshold;


cvx_end
logStatus{end+1}=cvx_status;
if ~strcmp(cvx_status,'Solved') || any(~isfinite([v1(:);v2(:);v3(:);vj(:)])), failed=true; break; end



v10=v1;v20=v2;v30=v3;vj0=vj;
t10=t1;t20=t2;t30=t3;e10=e1;e20=e2;e30=e3;
t_20=t_2;t_30=t_3;t_40=t_4;t_50=t_5;
sa10=sa1;sa20=sa2;sa30=sa3;


VJ1(k,i) = norm(vj)^2;
Rop1(k,i) = cvx_optval;
end



if failed, break; end
Hmr1 = diag(hr1)*Hsr;
Hmr2 = diag(hr2)*Hsr;
Hmr3 = diag(hr3)*Hsr;
Hmre = diag(hrt)*Hsr;

hr1v1 = [Hmr1 ; hs1']*v1;
hr1v2 = [Hmr1 ; hs1']*v2;
hr1v3 = [Hmr1 ; hs1']*v3;
hr1vj = [Hmr1 ; hs1']*vj;
hr2v1 = [Hmr2 ; hs2']*v1;
hr2v2 = [Hmr2 ; hs2']*v2;
hr2v3 = [Hmr2 ; hs2']*v3;
hr2vj = [Hmr2 ; hs2']*vj;
hr3v1 = [Hmr3 ; hs3']*v1;
hr3v2 = [Hmr3 ; hs3']*v2;
hr3v3 = [Hmr3 ; hs3']*v3;
hr3vj = [Hmr3 ; hs3']*vj;
hrev1 = [Hmre ; hst']*v1;
hrev2 = [Hmre ; hst']*v2;
hrev3 = [Hmre ; hst']*v3;
hrevj = [Hmre ; hst']*vj;






if isfield(cfg,'fixedIRS') && cfg.fixedIRS, continue; end
for j=1:cfg.inner
cvx_begin quiet
cvx_solver Mosek

variable V(N,1) complex
variables e1 e2 e3 t1 t2 t3 t_6
variables t_2 t_3 t_4 t_5 z_2 z_3 z_4 z_5
variables sa1 sa2 sa3
maximize t_6
subject to
t_6>=0;


abs(V)<=1;


t1>= 2^r1;
t2>= 2^r2;
t3>= 2^r3;


norm([2*t_6,(e3-z_5)]')<=e3+z_5;
z_5<=t_50^2+2*t_50*(t_5-t_50);
norm([2*t_5,(e2-z_4)]')<=e2+z_4;
z_4<=t_40^2+2*t_40*(t_4-t_40);
norm([2*t_4,(e1-z_3)]')<=e1+z_3;
z_3<=t_30^2+2*t_30*(t_3-t_30);
norm([2*t_2,(t1-t2)]')<=t1+t2;
norm([2*t_3,(t3-z_2)]')<=t3+z_2;
z_2<=t_20^2+2*t_20*(t_2-t_20);

g11=2*real([V0' 1]*hr1v1*hr1v1'*[V' 1]')/(t10-1)-real([V0' 1]*hr1v1*hr1v1'*[V0' 1]')*(t1-1)/(t10-1)^2-sigma;
g21=2*real([V0' 1]*hr2v1*hr2v1'*[V' 1]')/(t10-1)-real([V0' 1]*hr2v1*hr2v1'*[V0' 1]')*(t1-1)/(t10-1)^2-sigma;
g31=2*real([V0' 1]*hr3v1*hr3v1'*[V' 1]')/(t10-1)-real([V0' 1]*hr3v1*hr3v1'*[V0' 1]')*(t1-1)/(t10-1)^2-sigma;
norm([2*(hr1v2'*[V' 1]'),2*(hr1v3'*[V' 1]'),g11-1]')<=g11+1;
norm([2*(hr2v2'*[V' 1]'),2*(hr2v3'*[V' 1]'),g21-1]')<=g21+1;
norm([2*(hr3v2'*[V' 1]'),2*(hr3v3'*[V' 1]'),g31-1]')<=g31+1;

g22=2*real([V0' 1]*hr2v2*hr2v2'*[V' 1]')/(t20-1)-real([V0' 1]*hr2v2*hr2v2'*[V0' 1]')*(t2-1)/(t20-1)^2-sigma;
g32=2*real([V0' 1]*hr3v2*hr3v2'*[V' 1]')/(t20-1)-real([V0' 1]*hr3v2*hr3v2'*[V0' 1]')*(t2-1)/(t20-1)^2-sigma;
norm([2*(hr2v3'*[V' 1]'),g22-1]')<=g22+1;
norm([2*(hr3v3'*[V' 1]'),g32-1]')<=g32+1;

g33=2*real([V0' 1]*hr3v3*hr3v3'*[V' 1]')/(t30-1)-real([V0' 1]*hr3v3*hr3v3'*[V0' 1]')*(t3-1)/(t30-1)^2-sigma;
0<=g33;

q21=2*real([V0' 1]*hrev2*hrev2'*[V' 1]')/(e10)-real([V0' 1]*hrev2*hrev2'*[V0' 1]')*(e1)/(e10)^2;
q31=2*real([V0' 1]*hrev3*hrev3'*[V' 1]')/(e10)-real([V0' 1]*hrev3*hrev3'*[V0' 1]')*(e1)/(e10)^2;
qa1=2*real([V0' 1]*hrevj*hrevj'*[V' 1]')/(e10)-real([V0' 1]*hrevj*hrevj'*[V0' 1]')*(e1)/(e10)^2;
Q1=q21+q31+qa1  + 2*sigma/(e10)-sigma*(e1)/(e10)^2 - sigma;
norm([2*(hrev1'*[V' 1]'),2*(hrev2'*[V' 1]'),2*(hrev3'*[V' 1]'),2*(hrevj'*[V' 1]'),(Q1-1)]')<= Q1+1;

q12=2*real([V0' 1]*hrev1*hrev1'*[V' 1]')/(e20)-real([V0' 1]*hrev1*hrev1'*[V0' 1]')*(e2)/(e20)^2;
q32=2*real([V0' 1]*hrev3*hrev3'*[V' 1]')/(e20)-real([V0' 1]*hrev3*hrev3'*[V0' 1]')*(e2)/(e20)^2;
qa2=2*real([V0' 1]*hrevj*hrevj'*[V' 1]')/(e20)-real([V0' 1]*hrevj*hrevj'*[V0' 1]')*(e2)/(e20)^2;
Q2=q12+q32+qa2  + 2*sigma/(e20)-sigma*(e2)/(e20)^2 - sigma;
norm([2*(hrev1'*[V' 1]'),2*(hrev2'*[V' 1]'),2*(hrev3'*[V' 1]'),2*(hrevj'*[V' 1]'),(Q2-1)]')<= Q2+1;

q13=2*real([V0' 1]*hrev1*hrev1'*[V' 1]')/(e30)-real([V0' 1]*hrev1*hrev1'*[V0' 1]')*(e3)/(e30)^2;
q23=2*real([V0' 1]*hrev2*hrev2'*[V' 1]')/(e30)-real([V0' 1]*hrev2*hrev2'*[V0' 1]')*(e3)/(e30)^2;
qa3=2*real([V0' 1]*hrevj*hrevj'*[V' 1]')/(e30)-real([V0' 1]*hrevj*hrevj'*[V0' 1]')*(e3)/(e30)^2;
Q3=q13+q23+qa3  + 2*sigma/(e30)-sigma*(e3)/(e30)^2 - sigma;
norm([2*(hrev1'*[V' 1]'),2*(hrev2'*[V' 1]'),2*(hrev3'*[V' 1]'),2*(hrevj'*[V' 1]'),(Q3-1)]')<= Q3+1;


s1j=2*real([V0' 1]*hr1vj*hr1vj'*[V' 1]')-real([V0' 1]*hr1vj*hr1vj'*[V0' 1]');
s11=2*real([V0' 1]*hr1v1*hr1v1'*[V' 1]')-real([V0' 1]*hr1v1*hr1v1'*[V0' 1]');
s2j=2*real([V0' 1]*hr2vj*hr2vj'*[V' 1]')-real([V0' 1]*hr2vj*hr2vj'*[V0' 1]');
s21=2*real([V0' 1]*hr2v1*hr2v1'*[V' 1]')-real([V0' 1]*hr2v1*hr2v1'*[V0' 1]');
s22=2*real([V0' 1]*hr2v2*hr2v2'*[V' 1]')-real([V0' 1]*hr2v2*hr2v2'*[V0' 1]');
s3j=2*real([V0' 1]*hr3vj*hr3vj'*[V' 1]')-real([V0' 1]*hr3vj*hr3vj'*[V0' 1]');
s31=2*real([V0' 1]*hr3v1*hr3v1'*[V' 1]')-real([V0' 1]*hr3v1*hr3v1'*[V0' 1]');
s32=2*real([V0' 1]*hr3v2*hr3v2'*[V' 1]')-real([V0' 1]*hr3v2*hr3v2'*[V0' 1]');

norm(hr1v1'*[V' 1]')<=sqrt(s1j);
norm(hr1v2'*[V' 1]')<=sqrt(s11);
norm(hr1v3'*[V' 1]')<=sqrt(s11);
norm(hr2v1'*[V' 1]')<=sqrt(s2j);
norm(hr2v2'*[V' 1]')<=sqrt(s21);
norm(hr2v3'*[V' 1]')<=sqrt(s22);
norm(hr3v1'*[V' 1]')<=sqrt(s3j);
norm(hr3v2'*[V' 1]')<=sqrt(s31);
norm(hr3v3'*[V' 1]')<=sqrt(s32);


   La1=(2*real([V0' 1]*hr1vj*hr1vj'*[V' 1]')/(sa10-1)-real([V0' 1]*hr1vj*hr1vj'*[V0' 1]')*(sa1-1)/(sa10-1)^2)-sigma;
   La2=(2*real([V0' 1]*hr2vj*hr2vj'*[V' 1]')/(sa20-1)-real([V0' 1]*hr2vj*hr2vj'*[V0' 1]')*(sa2-1)/(sa20-1)^2)-sigma;
   La3=(2*real([V0' 1]*hr3vj*hr3vj'*[V' 1]')/(sa30-1)-real([V0' 1]*hr3vj*hr3vj'*[V0' 1]')*(sa3-1)/(sa30-1)^2)-sigma;

   norm([2*(hr1v2'*[V' 1]'),2*(hr1v3'*[V' 1]'),2*(hr1v1'*[V' 1]'),(La1-1)]')<=La1+1;
   norm([2*(hr2v2'*[V' 1]'),2*(hr2v3'*[V' 1]'),2*(hr2v1'*[V' 1]'),(La2-1)]')<=La2+1;
   norm([2*(hr3v2'*[V' 1]'),2*(hr3v3'*[V' 1]'),2*(hr3v1'*[V' 1]'),(La3-1)]')<=La3+1;

                sa1>=2^jamMin;sa2>=2^jamMin;sa3>=2^jamMin;


        he_ld=[Hmre;hst'];
        ldd1=4*real(v1'*he_ld'*[V0' 1]'*[V0' 1]*he_ld*he_ld'*[V' 1]'*[V0' 1]*he_ld*v1)-3*real(v1'*he_ld'*[V0' 1]'*[V0' 1]*he_ld*he_ld'*[V0' 1]'*[V0' 1]*he_ld*v1);
        ldd2=4*real(v2'*he_ld'*[V0' 1]'*[V0' 1]*he_ld*he_ld'*[V' 1]'*[V0' 1]*he_ld*v2)-3*real(v2'*he_ld'*[V0' 1]'*[V0' 1]*he_ld*he_ld'*[V0' 1]'*[V0' 1]*he_ld*v2);
        ldd3=4*real(v3'*he_ld'*[V0' 1]'*[V0' 1]*he_ld*he_ld'*[V' 1]'*[V0' 1]*he_ld*v3)-3*real(v3'*he_ld'*[V0' 1]'*[V0' 1]*he_ld*he_ld'*[V0' 1]'*[V0' 1]*he_ld*v3);
        ldd4=4*real(vj'*he_ld'*[V0' 1]'*[V0' 1]*he_ld*he_ld'*[V' 1]'*[V0' 1]*he_ld*vj)-3*real(vj'*he_ld'*[V0' 1]'*[V0' 1]*he_ld*he_ld'*[V0' 1]'*[V0' 1]*he_ld*vj);


echoFactor*(ldd1+ldd2+ldd3+ldd4) >= cfg.echoThreshold;
cvx_end
logStatus{end+1}=cvx_status;
if ~strcmp(cvx_status,'Solved') || any(~isfinite([v1(:);v2(:);v3(:);vj(:)])), failed=true; break; end



if any(~isfinite(V)), failed=true; break; end
V0=V;
t10=t1;t20=t2;t30=t3;e10=e1;e20=e2;e30=e3;
t_20=t_2;t_30=t_3;t_40=t_4;t_50=t_5;
sa10=sa1;sa20=sa2;sa30=sa3;
VJ2(k,j) = norm(vj0)^2;
Rop2(k,j) = cvx_optval;
probe.W=[v10 v20 v30 vj0];probe.V=exp(1i*angle(V0));
pm=evaluate_isac(ch,probe,cfg); if pm.feasible, last=probe; end
end
end

catch err
failed=true;errorMessage=string(err.message);cvx_clear;
end
candidate.W=[v10 v20 v30 vj0];candidate.V=exp(1i*angle(V0));
if failed, candidate=last; end
metrics=evaluate_isac(ch,candidate,cfg);
if ~failed && ~metrics.feasible && ~(isfield(cfg,'fixedIRS') && cfg.fixedIRS)
 fixcfg=cfg;fixcfg.fixedIRS=true;fixcfg.outer=1;
 [fixed,fixinfo]=solve_inherited(ch,fixcfg,candidate);
 if fixinfo.accepted, candidate=fixed;metrics=evaluate_isac(ch,candidate,cfg);repaired=true;end
end
info.runtime=toc(timer);info.status=logStatus;info.failed=failed;info.errorMessage=errorMessage;
info.metrics=metrics;info.accepted=~failed && metrics.feasible;info.repaired=repaired;
if info.accepted,sol=candidate;else,sol=last;end
info.returnedMetrics=evaluate_isac(ch,sol,cfg);
info.echoUpperBound=echoUpper;
end


