function [logpr, source] = extract_logpr_from_model(model)
logpr = NaN;
source = "";
candidates = collect_numeric_fields(model, "model");

keys = ["logevidence","logevid","logpr","marginal","evidence","loglike","likelihood","elbo","freeenergy","bound"];
best_score = -Inf;
for idx = 1:numel(candidates)
    name = lower(candidates(idx).name);
    x = candidates(idx).value;
    if isscalar(x) && isfinite(double(x))
        score = 0;
        for kk=1:numel(keys)
            if contains(name, keys(kk))
                score = score + 10 + kk;
            end
        end
        if score > best_score
            best_score = score;
            logpr = double(x);
            source = candidates(idx).name;
        end
    end
end

if ~isfinite(logpr)
    for idx = 1:numel(candidates)
        x = candidates(idx).value;
        if isscalar(x) && isfinite(double(x))
            logpr = double(x);
            source = candidates(idx).name + " [fallback scalar]";
            return;
        end
    end
end
end
