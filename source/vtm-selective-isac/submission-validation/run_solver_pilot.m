function run_solver_pilot
root=fileparts(mfilename('fullpath'));project=fileparts(fileparts(root));base=fullfile(project,'vtm-selective-isac');
addpath(base,fullfile(base,'normalized-10dB'),fullfile(base,'mobility-stress'),fullfile(base,'predictive-updates'));
c=predictive_config;out=fullfile(root,'initializer-pilot');files=dir(fullfile(out,'pred-*.mat'));rows={};
for f=1:numel(files)
 d=load(fullfile(out,files(f).name));[~,stem]=fileparts(d.source);parts=split(stem,'-');seed=str2double(parts{end});regime=parts{1};
 [traj,~,~]=mobility_trajectory(c,seed,regime);ref.V=ones(c.N,1);ref.W=ones(c.M,1)/sqrt(c.M)*sqrt(c.initialPowerFractions);
 for epoch=1:6
  ch=traj{20*(epoch-1)+1};sc=c;sc.rateMin=c.rateMin+c.optimizationRateMargin;sc.echoThreshold=c.echoThreshold*c.optimizationEchoFactor;sc.jamRateMin=1+c.optimizationJamMargin;sc.inner=1;sc.fixedIRS=true;
  % Rotate execution order to reduce systematic first-solver warmup bias.
  order=circshift(1:3,mod(epoch+f,3));
  for method=order
   timer=tic;fallback=false;inf=0;
   if method==1,[init,~]=mobility_initializer(ch,sc,ref);
   else
    ix=1;if method==2,ix=d.indices(epoch);inf=d.inference_s(epoch);end
    init=one_candidate(ch,sc,ref,ix);check=evaluate_isac(ch,init,sc);
    if ~check.feasible,[init,~]=mobility_initializer(ch,sc,ref);fallback=true;end
   end
   initTime=toc(timer)+inf;im=evaluate_isac(ch,init,sc);
   timer=tic;[sol,info]=solve_inherited(ch,sc,init);check=evaluate_isac(ch,sol,sc);solveTime=toc(timer);
   names={'scan','learned','simple'};
   rows(end+1,:)={stem,epoch,names{method},initTime,solveTime,initTime+solveTime,im.feasible,im.observableMargin,info.accepted && check.feasible,check.observableMargin,check.secrecy,fallback,info.failed};
  end
 end
 fprintf('SOLVER PILOT %s complete\n',stem);
end
T=cell2table(rows,'VariableNames',{'trajectory','epoch','method','init_s','solve_s','total_s','initial_feasible','initial_margin','accepted','returned_margin','secrecy','fallback','failed'});
writetable(T,fullfile(out,'solver-validation.csv'));
end

function candidate=one_candidate(ch,c,ref,ix)
candidate=ref;if ix==0,return;end
H=zeros(4,c.M);for u=1:4,H(u,:)=[ref.V' 1]*[diag(ch.hr(u,:))*ch.Hsr;ch.hs(:,u)'];end
if ix==1,q=sum(H'./max(vecnorm(H,2,2)',eps),2);
else,j=ix-1;q=pinv(H)*exp(1i*2*pi*(0:3)'*j/c.recoveryCodebookSize);end
if norm(q)>eps,candidate.W=q/norm(q)*sqrt(c.initialPowerFractions);end
end
