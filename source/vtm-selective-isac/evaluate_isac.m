function m=evaluate_isac(ch,s,cfg)
H=zeros(4,cfg.M);
for u=1:4, H(u,:)=[s.V' 1]*[diag(ch.hr(u,:))*ch.Hsr;ch.hs(:,u)']; end
A=abs(H*s.W).^2; noise=cfg.noiseW/cfg.powerW;
sensingNoise=cfg.noiseW;if isfield(cfg,'sensingNoiseW'),sensingNoise=cfg.sensingNoiseW;end
rates=zeros(1,3); decode=nan(3);
for k=1:3
 for u=k:3, decode(u,k)=log2(1+A(u,k)/(sum(A(u,k+1:3))+noise)); end
 rates(k)=min(decode(k:3,k));
end
eve=zeros(1,3);
for k=1:3, eve(k)=log2(1+A(4,k)/(sum(A(4,:))-A(4,k)+noise)); end
m.secrecy=sum(max(rates-eve,0));m.rates=rates;m.decode=decode;
m.echo=cfg.rcs*sum(abs(H(4,:)'*H(4,:)*s.W).^2,'all')/(sensingNoise/cfg.powerW);
m.powerFraction=sum(abs(s.W).^2,'all');m.unitError=max(abs(abs(s.V)-1));
m.jamRates=log2(1+A(1:3,4)./(sum(A(1:3,1:3),2)+noise));
jamMin=1;if isfield(cfg,'jamRateMin'),jamMin=cfg.jamRateMin;end
ordering=true;
for u=1:3
 tol=1e-8*max([noise A(u,:)]);
 ordering=ordering && A(u,4)>=A(u,1)-tol && A(u,1)>=A(u,2)-tol && A(u,2)>=A(u,3)-tol;
end
m.qosViolation=any(rates<cfg.rateMin-1e-5);
m.sensingViolation=m.echo<cfg.echoThreshold*(1-1e-5);
m.sicViolation=~ordering || any(m.jamRates<jamMin-1e-5) || any(sum(abs(s.W(:,1:3)).^2,1)>sum(abs(s.W(:,4)).^2)+1e-5);
m.feasible=all(isfinite(s.W),'all') && all(isfinite(s.V)) && m.powerFraction<=1+1e-5 && m.unitError<=1e-6 && ~m.qosViolation && ~m.sensingViolation && ~m.sicViolation;
m.qosMargin=min(rates)/cfg.rateMin-1;
m.echoMargin=m.echo/cfg.echoThreshold-1;
m.jamMargin=min(m.jamRates)/jamMin-1;
m.observableMargin=min([m.qosMargin m.echoMargin m.jamMargin]);
end
