function out = collect_numeric_fields(s, prefix)
out = struct('name', {}, 'value', {});
if isnumeric(s) || islogical(s)
    out(1).name = prefix;
    out(1).value = s;
    return;
end
if iscell(s)
    for i=1:numel(s)
        tmp = collect_numeric_fields(s{i}, prefix + "{" + string(i) + "}");
        out = [out, tmp]; %#ok<AGROW>
    end
    return;
end
if isstruct(s)
    f = fieldnames(s);
    for i=1:numel(f)
        tmp = collect_numeric_fields(s.(f{i}), prefix + "." + string(f{i}));
        out = [out, tmp]; %#ok<AGROW>
    end
end
end
