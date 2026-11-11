function res = fit_one_wsbm_official(A, k, n_internal_trials, seed, cfg)

rng(seed, 'twister');
N = size(A,1);
res = struct('labels', [], 'logpr', NaN, 'model', [], 'info', [], 'status', "", 'message', "");

if isempty(which('wsbm'))
    error('Model input or output validation failed.');
end

opts = struct();
opts.numTrials = n_internal_trials;
opts.verbosity = 0;
opts.seed = seed;

[i,j,w] = find(triu(A,1));
Edg = [i(:), j(:), w(:)];

lastErr = [];
model = [];
call_used = "";
stdout_txt = "";

try
    stdout_txt = evalc('model = wsbm(A, k, opts);');
    call_used = "wsbm(A,k,opts)";
catch ME
    lastErr = ME;
    model = [];
end

if isempty(model)
    try
        stdout_txt = evalc('model = wsbm(A, k);');
        call_used = "wsbm(A,k)";
    catch ME
        lastErr = ME;
        model = [];
    end
end

if isempty(model)
    try
        stdout_txt = evalc('model = wsbm(Edg, k, opts);');
        call_used = "wsbm(edge_list,k,opts)";
    catch ME
        lastErr = ME;
        model = [];
    end
end

if isempty(model)
    try
        stdout_txt = evalc('model = wsbm(Edg, k);');
        call_used = "wsbm(edge_list,k)";
    catch ME
        lastErr = ME;
        model = [];
    end
end

if isempty(model)
    error('Model input or output validation failed.', lastErr.message);
end

[labels, label_source] = extract_labels_from_model(model, N, k);
[logpr, logpr_source] = extract_logpr_from_model(model);

if ~isfinite(logpr)
    [logpr2, source2] = parse_logpr_from_wsbm_stdout(stdout_txt);
    if isfinite(logpr2)
        logpr = logpr2;
        logpr_source = source2;
    end
end

if (isempty(labels) || any(~isfinite(labels)) || numel(labels) ~= N || ~isfinite(logpr))
    try
        debug_dir = fullfile(cfg.OUT_DIR, 'debug_failed_wsbm_model');
        if ~exist(debug_dir, 'dir'); mkdir(debug_dir); end
        tag = sprintf('k%02d_seed%d', k, seed);
        dump_model_structure_text(model, fullfile(debug_dir, [tag '_model_structure.txt']));
        fid = fopen(fullfile(debug_dir, [tag '_stdout.txt']), 'w');
        if fid >= 0
            fprintf(fid, '%s', stdout_txt);
            fclose(fid);
        end
    catch
    end
end

if isempty(labels) || any(~isfinite(labels)) || numel(labels) ~= N
    error('Model input or output validation failed.');
end
if ~isfinite(logpr)
    error('Model input or output validation failed.');
end

labels = double(labels(:));
[~,~,labels] = unique(labels, 'stable');

res.labels = labels;
res.logpr = double(logpr);
res.model = [];
res.info = struct('call_used', call_used, 'label_source', label_source, 'logpr_source', logpr_source);
res.status = "OK";
res.message = "";
end
