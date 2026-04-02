<<<<<<< HEAD
function result = Normalization(GasSeq)
    % Normalize the gas sequence to zero mean and unit variance (z-score normalization)
    GasMean = mean(GasSeq);
    GasVar = var(GasSeq);
    result = (GasSeq - GasMean) / sqrt(GasVar);
end
=======
function result = Normalization(GasSeq)
    % Normalize the gas sequence to zero mean and unit variance (z-score normalization)
    GasMean = mean(GasSeq);
    GasVar = var(GasSeq);
    result = (GasSeq - GasMean) / sqrt(GasVar);
end
>>>>>>> 3c564e218f21b381bb86fa936c53f9a10b865e73
