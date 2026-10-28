function export_submission_assets
group=1;split='test';
% Isolated candidate library; no historical file is modified.
root=fileparts(mfilename('fullpath'));project=fileparts(fileparts(root));
addpath(fullfile(project,'vtm-selective-isac'),fullfile(project,'vtm-selective-isac','normalized-10dB'), ...
 fullfile(project,'vtm-selective-isac','mobility-stress'),fullfile(project,'vtm-selective-isac','predictive-updates'));
c=predictive_config;c.directionTimes=0:5:115;
seeds=16101:16124;regimes=c.testRegimes;
out=fullfile(root,'assets',split);if ~isfolder(out),mkdir(out);end
for group=1:3
for seed=seeds
 target=fullfile(out,sprintf('%s-%d.mat',regimes{group},seed));if isfile(target),fprintf('Cached %s\n',target);continue;end
 [traj,probes,meta]=mobility_trajectory(c,seed,regimes{group});z=latent(c,seed);
 ref.V=ones(c.N,1);ref.W=ones(c.M,1)/sqrt(c.M)*sqrt(c.initialPowerFractions);
 setupTimer=tic;[initial,ok]=mobility_initializer(traj{1},c,ref);setupTime=toc(setupTimer);assert(ok,'Keep infeasible starts visible');
 n=numel(c.directionTimes);Ws=zeros(c.M,4,n,2);Vs=zeros(c.N,n,2);accepted=false(n,2);failed=accepted;runtime=zeros(n,2);
 for k=1:n
  for a=1:2,Ws(:,:,k,a)=initial.W;Vs(:,k,a)=initial.V;end
  if mod(k-1,4)~=0,continue;end
  ch=traj{c.directionTimes(k)+1};sc=c;sc.rateMin=c.rateMin+c.optimizationRateMargin;sc.echoThreshold=c.echoThreshold*c.optimizationEchoFactor;sc.jamRateMin=1+c.optimizationJamMargin;
  sc.inner=1;sc.fixedIRS=true;
  timer=tic;[init,~]=mobility_initializer(ch,sc,ref);
  [candidate,info]=solve_inherited(ch,sc,init);
  check=evaluate_isac(ch,candidate,sc);runtime(k,1)=toc(timer);
  Ws(:,:,k,1)=candidate.W;Vs(:,k,1)=candidate.V;accepted(k,1)=info.accepted && check.feasible;failed(k,1)=info.failed;
 end
 testW=initial.W;testV=initial.V;testMetrics=zeros(5,5);ix=[1 21 41 81 121];
 for j=1:5,m=evaluate_isac(traj{ix(j)},initial,c);testMetrics(j,:)=[m.secrecy m.feasible m.echo m.rates(1) m.observableMargin];end
 save(target,'c','z','probes','meta','initial','Ws','Vs','accepted','failed','runtime','testW','testV','testMetrics','setupTime','-v7');
 fprintf('ASSET COMPLETE %s\n',target);
end
end
end

function z=latent(c,seed)
rng(seed,'twister');rand;randi(12);randi(10);angles=2*pi*rand(c.scatterers,4);z.phases=2*pi*rand(c.scatterers,4);
z.coef=(randn(c.M,c.scatterers,4)+1i*randn(c.M,c.scatterers,4))/sqrt(2*c.scatterers*c.M);
z.coefIRS=(randn(c.scatterers,c.N,4)+1i*randn(c.scatterers,c.N,4))/sqrt(2*c.scatterers*c.N);
z.dx=cos(angles);z.dy=sin(angles);lambda=c.lightSpeed/c.carrierHz;
steer=@(n,d)exp(1i*pi*(0:n-1)'*(d(2)/norm(d)))/sqrt(n);
z.Hsr=.12*steer(c.N,c.bsPosition-c.irsPosition)*steer(c.M,c.irsPosition-c.bsPosition)'*exp(-1i*2*pi*norm(c.irsPosition-c.bsPosition)/lambda);
end

