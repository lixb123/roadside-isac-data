function c=predictive_config
c=mobility_config;
c.testSeeds=7101:7106; % fresh paired evaluation, not the inspected prior test seeds
% Fixed before any new validation/test result is examined.
c.predictionHorizonSeconds=5;c.historySeconds=4;
c.positionNoiseStdM=.20;c.telemetrySeedOffset=930000;
c.probeFraction=.04; % extra historical power pilots and position telemetry
c.forecastMarginThreshold=.08;
c.featureNames=[c.featureNames {'angular_rate_rad_s','forecast_array_loss', ...
 'estimated_blockage_db','power_drop_db_s','forecast_qos_proxy'}];
c.description='Causal paid noisy telemetry and pilot histories; 5s constant-velocity/angular prediction. Same normalized propagation and snapshot limits as mobility-stress.';
end
