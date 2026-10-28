function c=benchmark_config
% Predeclared assumptions, never inferred from policy winners.
c=struct('M',6,'N',8,'powerW',1,'noiseW',.005,'sensingNoiseW',.005, ...
 'rcs',1,'echoThreshold',10,'rateMin',1,'outer',1,'inner',2,'initializeAux',true);
c.slotSeconds=20;c.slots=6;c.rho=.995;c.innovation=.10;
c.probeFraction=.02;c.acquisitionFraction=.03;c.acquisitionDelay=.2;
c.probeNoise=.01;c.maxAgeSeconds=40;c.riskMargin=.08;
c.optimizationRateMargin=.10;c.optimizationEchoFactor=1.20;c.optimizationJamMargin=.10;
c.periodSeconds=40;c.residualThreshold=.08;
c.trainSeeds=1101:1116;c.validationSeeds=2101:2104;c.testSeeds=3101:3106;
c.sensitivityRho=[.999 .995 .98];c.sensitivitySensingNoise=[.0025 .005 .01];
c.sensitivityDelay=[.2 2 5];
c.featureNames={'age_seconds','probe_residual','estimated_qos_margin','estimated_echo_margin','estimated_jam_margin'};
c.description='Dimensionless correlated rank-one benchmark; no physical air-ground calibration.';
end
