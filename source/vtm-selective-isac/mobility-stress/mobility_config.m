function c=mobility_config
c=benchmark_config;
c.trainSeeds=4101:4116;c.validationSeeds=5101:5104;c.testSeeds=6101:6106;
c.carrierHz=2.4e9;c.lightSpeed=299792458;c.referenceDistance=100;
c.bsPosition=[0 0 10];c.irsPosition=[20 -20 15];
c.initialPositions=[140 0 40;110 0 1.5;90 0 1.5;150 0 35];
c.ricianKdB=25;c.scatterers=24;c.scoreStepSeconds=1;
c.nominalSpeed=2;c.speedSpread=.25;c.blockageDb=8;c.blockageDuration=16;
c.periodSeconds=40;c.maxAgeSeconds=40;
c.description='Geometry-aware normalized reference budget; free-space attenuation, ULA steering, path phase and SOS Doppler. Snapshot-level assessment, no calibrated echo receiver.';
c.trainRegimes={'cruise','turn_blockage'};c.testRegimes={'cruise','turn_blockage','ood'};
c.oodSpeed=5;c.oodBlockageDb=12;c.oodBlockageDuration=24;
c.initialPowerFractions=[.27 .12 .05 .56];
c.recoveryCodebookSize=16;
c.beamResidualThreshold=.15;
c.featureNames=[c.featureNames {'paid_beam_power_residual'}];
end
