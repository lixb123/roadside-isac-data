function [best,feasible]=mobility_initializer(ch,c,old)
% Uses ONLY already acquired CSI. Same deterministic restart for all policies.
H=zeros(4,c.M);
for u=1:4,H(u,:)=[old.V' 1]*[diag(ch.hr(u,:))*ch.Hsr;ch.hs(:,u)'];end
best=old;m=evaluate_isac(ch,best,c);feasible=m.feasible;bestMargin=m.observableMargin;
for j=0:c.recoveryCodebookSize
 if j==0,q=sum(H'./max(vecnorm(H,2,2)',eps),2);
 else,ph=exp(1i*2*pi*(0:3)'*j/c.recoveryCodebookSize);q=pinv(H)*ph;end
 if norm(q)<eps,continue;end
 candidate.V=old.V;candidate.W=q/norm(q)*sqrt(c.initialPowerFractions);
 cm=evaluate_isac(ch,candidate,c);
 if (cm.feasible && ~feasible) || (cm.feasible==feasible && cm.observableMargin>bestMargin)
  best=candidate;feasible=cm.feasible;bestMargin=cm.observableMargin;
 end
end
end
