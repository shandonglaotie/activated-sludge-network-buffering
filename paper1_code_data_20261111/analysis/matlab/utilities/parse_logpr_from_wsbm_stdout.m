function [logpr, source] = parse_logpr_from_wsbm_stdout(txt)

logpr = NaN;
source = "";
if isempty(txt)
    return;
end

pat_best = 'Best:\s*([-+]?\d*\.?\d+(?:[eEdD][-+]?\d+)?)';
tokens = regexp(txt, pat_best, 'tokens');
if ~isempty(tokens)
    vals = nan(numel(tokens), 1);
    for i = 1:numel(tokens)
        s = strrep(tokens{i}{1}, 'D', 'E');
        s = strrep(s, 'd', 'e');
        vals(i) = str2double(s);
    end
    vals = vals(isfinite(vals));
    if ~isempty(vals)
        logpr = vals(end);
        source = "stdout_last_Best";
        return;
    end
end

pat_trial = 'Trial\s+\d+\s+of\s+\d+\s*:\s*([-+]?\d*\.?\d+(?:[eEdD][-+]?\d+)?)';
tokens = regexp(txt, pat_trial, 'tokens');
if ~isempty(tokens)
    vals = nan(numel(tokens), 1);
    for i = 1:numel(tokens)
        s = strrep(tokens{i}{1}, 'D', 'E');
        s = strrep(s, 'd', 'e');
        vals(i) = str2double(s);
    end
    vals = vals(isfinite(vals));
    if ~isempty(vals)
        logpr = max(vals);
        source = "stdout_max_trial_value";
        return;
    end
end
end
