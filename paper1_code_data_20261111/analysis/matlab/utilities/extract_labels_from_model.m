function [labels, source] = extract_labels_from_model(model, N, k)
labels = [];
source = "";

candidates = collect_numeric_fields(model, "model");

for idx = 1:numel(candidates)
    x = candidates(idx).value;
    if isvector(x) && numel(x)==N
        xv = double(x(:));
        if all(isfinite(xv)) && numel(unique(xv)) <= max(k,2) && all(abs(xv-round(xv)) < 1e-8)
            labels = round(xv);
            source = candidates(idx).name;
            return;
        end
    end
end

for idx = 1:numel(candidates)
    x = double(candidates(idx).value);
    if ismatrix(x)
        [r,c] = size(x);
        if r==N && c>=2 && c<=max(k+5,2)
            [~, labels] = max(x, [], 2);
            source = candidates(idx).name + " -> max(row)";
            return;
        elseif c==N && r>=2 && r<=max(k+5,2)
            [~, labels] = max(x, [], 1);
            labels = labels(:);
            source = candidates(idx).name + " -> max(col)";
            return;
        end
    end
end
end
