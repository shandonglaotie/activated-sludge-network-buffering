function dump_model_structure_text(model, out_file)
try
    fid = fopen(out_file, 'w');
    if fid < 0
        return;
    end
    fprintf(fid, 'WSBM model class: %s\n', class(model));
    fprintf(fid, 'WSBM model size : %s\n\n', mat2str(size(model)));
    dump_one(fid, model, 'model', 0);
    fclose(fid);
catch
    try
        fclose(fid);
    catch
    end
end
end

function dump_one(fid, x, name, depth)
indent = repmat('  ', 1, depth);
if depth > 6
    fprintf(fid, '%s%s : <max depth reached>\n', indent, name);
    return;
end

if isnumeric(x) || islogical(x)
    fprintf(fid, '%s%s : %s %s', indent, name, class(x), mat2str(size(x)));
    if isscalar(x)
        fprintf(fid, ' value=%.12g', double(x));
    elseif numel(x) <= 12
        try
            fprintf(fid, ' values=%s', mat2str(double(x(:))'));
        catch
        end
    end
    fprintf(fid, '\n');
elseif ischar(x) || isstring(x)
    fprintf(fid, '%s%s : %s %s\n', indent, name, class(x), mat2str(size(x)));
elseif iscell(x)
    fprintf(fid, '%s%s : cell %s\n', indent, name, mat2str(size(x)));
    n = min(numel(x), 10);
    for i = 1:n
        dump_one(fid, x{i}, sprintf('%s{%d}', name, i), depth+1);
    end
    if numel(x) > n
        fprintf(fid, '%s  ... %d more cells\n', indent, numel(x)-n);
    end
elseif isstruct(x)
    fprintf(fid, '%s%s : struct %s\n', indent, name, mat2str(size(x)));
    if numel(x) > 1
        x = x(1);
        fprintf(fid, '%s  only first struct element shown\n', indent);
    end
    f = fieldnames(x);
    for i = 1:numel(f)
        dump_one(fid, x.(f{i}), sprintf('%s.%s', name, f{i}), depth+1);
    end
else
    fprintf(fid, '%s%s : %s %s\n', indent, name, class(x), mat2str(size(x)));
end
end
