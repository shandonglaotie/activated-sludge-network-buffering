function run_wsbm(k_values, repeats)
if nargin < 1; k_values = 5:20; end
if nargin < 2; repeats = 5; end
here = fileparts(mfilename('fullpath'));
root = fileparts(fileparts(here));
addpath(genpath(fullfile(here,'wsbm')));
addpath(fullfile(here,'utilities'));
cfg.OUT_DIR = fullfile(root,'results','wsbm');
if ~exist(cfg.OUT_DIR,'dir'); mkdir(cfg.OUT_DIR); end
input = load(fullfile(root,'figures','supplementary','figure_s01','data','wsbm_input_signed_adjacency.mat'),'A','node_ids');
A = input.A;
rows = {}; labels_all = struct;
for k = k_values
    for repeat = 1:repeats
        seed = 20260706 + k*10000 + repeat;
        res = fit_one_wsbm_official(A,k,5,seed,cfg);
        rows(end+1,:) = {k,repeat,seed,res.logpr,res.status,res.message};
        labels_all.(sprintf('k%02d_rep%03d',k,repeat)) = res.labels;
        fprintf('k=%d repeat=%d logpr=%.12g\n',k,repeat,res.logpr);
    end
end
scan = cell2table(rows,'VariableNames',{'k','repeat','seed','logpr','status','message'});
writetable(scan,fullfile(cfg.OUT_DIR,'wsbm_logpr.csv'));
save(fullfile(cfg.OUT_DIR,'wsbm_labels.mat'),'labels_all','-v7');
end
