<<<<<<< HEAD
function [t, V, I, Q, CO, CO2, C2H4] = ProcessCycleData(cycleInfo)
    % Extract time, voltage, current, and capacity from cycle structure
    t = cycleInfo.Time;
    V = cycleInfo.Voltage;
    I = cycleInfo.Current;
    Q = cycleInfo.Capacity;
    
    % Get state-of-charge index for gas sampling
    SOC_idx = cycleInfo.SOC_idx;

    % Sample gas data (CO, CO2, C2H4) at the SOC indices
    CO = SampleGas(cycleInfo.CO, SOC_idx)';
    CO2 = SampleGas(cycleInfo.CO2, SOC_idx)';
    C2H4 = SampleGas(cycleInfo.C2H4, SOC_idx)';
=======
function [t, V, I, Q, CO, CO2, C2H4] = ProcessCycleData(cycleInfo)
    % Extract time, voltage, current, and capacity from cycle structure
    t = cycleInfo.Time;
    V = cycleInfo.Voltage;
    I = cycleInfo.Current;
    Q = cycleInfo.Capacity;
    
    % Get state-of-charge index for gas sampling
    SOC_idx = cycleInfo.SOC_idx;

    % Sample gas data (CO, CO2, C2H4) at the SOC indices
    CO = SampleGas(cycleInfo.CO, SOC_idx)';
    CO2 = SampleGas(cycleInfo.CO2, SOC_idx)';
    C2H4 = SampleGas(cycleInfo.C2H4, SOC_idx)';
>>>>>>> 3c564e218f21b381bb86fa936c53f9a10b865e73
end