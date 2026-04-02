<<<<<<< HEAD
function result = OutliersRemoval(sequence, window_size, n)
    % Remove outliers using a moving window: replace values outside mean ± n*std with the window mean
    seq_len = length(sequence);

    for i = 1+window_size/2:seq_len-window_size/2
        seq_cut = sequence(i-window_size/2:i+window_size/2);
        mu = mean(seq_cut);
        sigma = std(seq_cut);
    
        if sequence(i) <= mu - n*sigma || sequence(i) >= mu + n*sigma
            sequence(i) = mu;
        end
    end

    result = sequence;
end
=======
function result = OutliersRemoval(sequence, window_size, n)
    % Remove outliers using a moving window: replace values outside mean ± n*std with the window mean
    seq_len = length(sequence);

    for i = 1+window_size/2:seq_len-window_size/2
        seq_cut = sequence(i-window_size/2:i+window_size/2);
        mu = mean(seq_cut);
        sigma = std(seq_cut);
    
        if sequence(i) <= mu - n*sigma || sequence(i) >= mu + n*sigma
            sequence(i) = mu;
        end
    end

    result = sequence;
end
>>>>>>> 3c564e218f21b381bb86fa936c53f9a10b865e73
