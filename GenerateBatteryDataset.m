<<<<<<< HEAD
function batch = GenerateBatteryDataset()
    %% 1. Set paths
    dataFolder = 'processed_data';
    excelFile = 'dataset\CellRetention.xlsx';
    
    if ~exist(dataFolder, 'dir')
        error('Data folder "%s" does not exist', dataFolder);
    end
    
    if ~exist(excelFile, 'file')
        warning('Excel file "%s" does not exist', excelFile);
    end
    
    %% 2. Get battery files
    filePattern = fullfile(dataFolder, 'Cell*_pro.mat');
    files = dir(filePattern);
    
    if isempty(files)
        warning('No matching files found');
        batch = struct();
        return;
    end
    
    %% 3. Process batteries
    batteryDataCell = cell(length(files), 1);
    
    for i = 1:length(files)
        filename = files(i).name;
        pattern = 'cell(.*)_pro\.mat';
        matches = regexp(filename, pattern, 'tokens', 'once');
        if isempty(matches)
            warning('Cannot extract battery ID from filename "%s"', filename);
            batteryDataCell{i} = [];
            continue;
        end
        cell_id = string(matches{1});
        
        % Load data
        try
            data = load(fullfile(dataFolder, filename));
            if ~isfield(data, 'cycleData') || ~isstruct(data.cycleData)
                error('Missing cycleData structure');
            end
        catch ME
            warning('Failed to load %s: %s', filename, ME.message);
            batteryDataCell{i} = [];
            continue;
        end
        
        % Read Excel data
        try
            sheetName = strcat('Cell', cell_id);
            rawTable = readtable(excelFile, 'Sheet', sheetName, 'HeaderLines', 1);
            summaryData = struct('cycle', rawTable.Var1', 'QDischarge', rawTable.Var2');
        catch ME
            warning('Failed to read Excel for %s: %s', cell_id, ME.message);
            batteryDataCell{i} = [];
            continue;
        end
        
        % Process battery
        try
            singleBatch = ProcessSingleBattery(data.cycleData, cell_id, summaryData);
            batteryDataCell{i} = singleBatch;
        catch ME
            warning('Processing failed for %s: %s', cell_id, ME.message);
            batteryDataCell{i} = [];
        end
    end
    
    %% 4. Combine results
    batteryDataCell = batteryDataCell(~cellfun('isempty', batteryDataCell));
    batch = struct();
    if ~isempty(batteryDataCell)
        batch = [batteryDataCell{:}];
        fprintf('Processing complete! Total batteries: %d\n', length(batch));
    end
=======
function batch = GenerateBatteryDataset()
    %% 1. Set paths
    dataFolder = 'processed_data';
    excelFile = 'dataset\CellRetention.xlsx';
    
    if ~exist(dataFolder, 'dir')
        error('Data folder "%s" does not exist', dataFolder);
    end
    
    if ~exist(excelFile, 'file')
        warning('Excel file "%s" does not exist', excelFile);
    end
    
    %% 2. Get battery files
    filePattern = fullfile(dataFolder, 'Cell*_pro.mat');
    files = dir(filePattern);
    
    if isempty(files)
        warning('No matching files found');
        batch = struct();
        return;
    end
    
    %% 3. Process batteries
    batteryDataCell = cell(length(files), 1);
    
    for i = 1:length(files)
        filename = files(i).name;
        pattern = 'cell(.*)_pro\.mat';
        matches = regexp(filename, pattern, 'tokens', 'once');
        if isempty(matches)
            warning('Cannot extract battery ID from filename "%s"', filename);
            batteryDataCell{i} = [];
            continue;
        end
        cell_id = string(matches{1});
        
        % Load data
        try
            data = load(fullfile(dataFolder, filename));
            if ~isfield(data, 'cycleData') || ~isstruct(data.cycleData)
                error('Missing cycleData structure');
            end
        catch ME
            warning('Failed to load %s: %s', filename, ME.message);
            batteryDataCell{i} = [];
            continue;
        end
        
        % Read Excel data
        try
            sheetName = strcat('Cell', cell_id);
            rawTable = readtable(excelFile, 'Sheet', sheetName, 'HeaderLines', 1);
            summaryData = struct('cycle', rawTable.Var1', 'QDischarge', rawTable.Var2');
        catch ME
            warning('Failed to read Excel for %s: %s', cell_id, ME.message);
            batteryDataCell{i} = [];
            continue;
        end
        
        % Process battery
        try
            singleBatch = ProcessSingleBattery(data.cycleData, cell_id, summaryData);
            batteryDataCell{i} = singleBatch;
        catch ME
            warning('Processing failed for %s: %s', cell_id, ME.message);
            batteryDataCell{i} = [];
        end
    end
    
    %% 4. Combine results
    batteryDataCell = batteryDataCell(~cellfun('isempty', batteryDataCell));
    batch = struct();
    if ~isempty(batteryDataCell)
        batch = [batteryDataCell{:}];
        fprintf('Processing complete! Total batteries: %d\n', length(batch));
    end
>>>>>>> 3c564e218f21b381bb86fa936c53f9a10b865e73
end