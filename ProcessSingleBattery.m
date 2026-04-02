<<<<<<< HEAD
function singleBatch = ProcessSingleBattery(cycleData, cell_id, summaryData)
    singleBatch.policy = '';
    singleBatch.policy_readable = '';
    singleBatch.channel_id = cell_id;
    singleBatch.cycle_life = NaN;
    singleBatch.summary = summaryData;
    
    % Identify cycle fields
    allFields = fieldnames(cycleData);
    cyclesCell = cell(length(allFields), 1);
    
    for j = 1:length(allFields)
        fieldName = allFields{j};
        if ~isstruct(cycleData.(fieldName))
            continue;
        end
        
        % Check required fields
        required = {'Time', 'Voltage', 'Current', 'Capacity', 'SOC_idx', 'CO', 'CO2', 'C2H4'};
        if ~all(ismember(required, fieldnames(cycleData.(fieldName))))
            continue;
        end
        
        % Extract cycle number
        try
            cycleNum = str2double(regexp(fieldName, '\d+', 'match', 'once'));
        catch
            continue;
        end
        
        % Process data
        try
            cyc = cycleData.(fieldName);
            [t, V, I, Q, CO, CO2, C2H4] = ProcessCycleData(cyc);
            
            cyclesCell{j} = struct('cycle', cycleNum, 't', t, 'V', V, 'I', I, ...
                                   'Q', Q, 'CO', CO, 'CO2', CO2, 'C2H4', C2H4);
        catch
            cyclesCell{j} = [];
        end
    end
    
    % Sort and save
    cyclesCell = cyclesCell(~cellfun('isempty', cyclesCell));
    if ~isempty(cyclesCell)
        cyclesList = [cyclesCell{:}];
        [~, sortIdx] = sort([cyclesList.cycle]);
        singleBatch.cycles = cyclesList(sortIdx);
    else
        singleBatch.cycles = struct();
    end
=======
function singleBatch = ProcessSingleBattery(cycleData, cell_id, summaryData)
    singleBatch.policy = '';
    singleBatch.policy_readable = '';
    singleBatch.channel_id = cell_id;
    singleBatch.cycle_life = NaN;
    singleBatch.summary = summaryData;
    
    % Identify cycle fields
    allFields = fieldnames(cycleData);
    cyclesCell = cell(length(allFields), 1);
    
    for j = 1:length(allFields)
        fieldName = allFields{j};
        if ~isstruct(cycleData.(fieldName))
            continue;
        end
        
        % Check required fields
        required = {'Time', 'Voltage', 'Current', 'Capacity', 'SOC_idx', 'CO', 'CO2', 'C2H4'};
        if ~all(ismember(required, fieldnames(cycleData.(fieldName))))
            continue;
        end
        
        % Extract cycle number
        try
            cycleNum = str2double(regexp(fieldName, '\d+', 'match', 'once'));
        catch
            continue;
        end
        
        % Process data
        try
            cyc = cycleData.(fieldName);
            [t, V, I, Q, CO, CO2, C2H4] = ProcessCycleData(cyc);
            
            cyclesCell{j} = struct('cycle', cycleNum, 't', t, 'V', V, 'I', I, ...
                                   'Q', Q, 'CO', CO, 'CO2', CO2, 'C2H4', C2H4);
        catch
            cyclesCell{j} = [];
        end
    end
    
    % Sort and save
    cyclesCell = cyclesCell(~cellfun('isempty', cyclesCell));
    if ~isempty(cyclesCell)
        cyclesList = [cyclesCell{:}];
        [~, sortIdx] = sort([cyclesList.cycle]);
        singleBatch.cycles = cyclesList(sortIdx);
    else
        singleBatch.cycles = struct();
    end
>>>>>>> 3c564e218f21b381bb86fa936c53f9a10b865e73
end