<<<<<<< HEAD
function sampled = SampleGas(gasData, SOC_idx)
    % Initialize output as empty
    sampled = [];
    
    % Return empty if gas data or SOC index array is empty
    if isempty(gasData) || isempty(SOC_idx)
        return;
    end
    
    % Additional check for empty SOC_idx (already covered above, but kept for clarity)
    if isempty(SOC_idx)
        return;
    end
    
    % Number of sampling intervals
    n = length(SOC_idx);
    sampled = zeros(1, n);
    
    % Average gas data over each segment defined by SOC indices
    for i = 1:n
        startIdx = SOC_idx(i);
        if i < n
            endIdx = SOC_idx(i+1) - 1;
        else
            endIdx = length(gasData);
        end
        sampled(i) = mean(gasData(startIdx:endIdx));
    end
=======
function sampled = SampleGas(gasData, SOC_idx)
    % Initialize output as empty
    sampled = [];
    
    % Return empty if gas data or SOC index array is empty
    if isempty(gasData) || isempty(SOC_idx)
        return;
    end
    
    % Additional check for empty SOC_idx (already covered above, but kept for clarity)
    if isempty(SOC_idx)
        return;
    end
    
    % Number of sampling intervals
    n = length(SOC_idx);
    sampled = zeros(1, n);
    
    % Average gas data over each segment defined by SOC indices
    for i = 1:n
        startIdx = SOC_idx(i);
        if i < n
            endIdx = SOC_idx(i+1) - 1;
        else
            endIdx = length(gasData);
        end
        sampled(i) = mean(gasData(startIdx:endIdx));
    end
>>>>>>> 3c564e218f21b381bb86fa936c53f9a10b865e73
end