clear;close all;clc;

task = 'IDH';   %% IDH / MGMT

data_dir = fullfile(pwd, 'Data', [task, '_Example']);
Z = load(data_dir).Z;                  
Background = load(data_dir).Background;
Tumor_mask = load(data_dir).Tumor_mask;

model_dir = fullfile(pwd, [task, '_net'], 'FCN_NET');
S = load(model_dir).net;

colormap_dir = fullfile(pwd, 'Data','Probability_Colormap');
Probability_Colormap = load(colormap_dir).Probability_Colormap;

model = extractNetObject(S);
assert(~isempty(model) && ismethod(model, 'classify'), 'Model not available: No classifying network object found');

[H, W, ~] = size(Z);
mask = logical(Tumor_mask);
assert(all(size(mask) == [H, W]), 'Tumor_mask size mismatch');

Zmat = reshape(Z, H * W, 61);
X_tumor = Zmat(mask(:), :);

[YPred, scores] = model.classify(X_tumor);

[probMap, subtype, p] = classifyTumorType(scores, task, Z, mask);

figure;
showOverlayProb(Background, probMap, mask, subtype, p, Probability_Colormap);

function mdl = extractNetObject(s)
    mdl = [];
    if isa(s, 'SeriesNetwork') || isa(s, 'DAGNetwork') || isa(s, 'dlnetwork')
        mdl = s; return;
    end
    if isstruct(s)
        fns = fieldnames(s);
        for i = 1:numel(fns)
            mdl = extractNetObject(s.(fns{i}));
            if ~isempty(mdl), return; end
        end
    elseif iscell(s)
        for i = 1:numel(s)
            mdl = extractNetObject(s{i});
            if ~isempty(mdl), return; end
        end
    end
end

function [probMap, subtype, p] = classifyTumorType(scores, task, Z_for_pred, mask)

    avg_scores = mean(scores, 1);

    p = [];

    switch lower(task)
        case 'idh'
            if avg_scores(1) > avg_scores(2)
                subtype = 'IDH-wt';
                p = scores(:, 1);
            elseif avg_scores(1) < avg_scores(2)
                subtype = 'IDH-mut';
                p = scores(:, 2);
            else
                subtype = 'Unknown';
                p = zeros(size(scores, 1), 1);
            end
        case 'mgmt'
            if avg_scores(1) > avg_scores(2)
                subtype = 'MGMT-unmet';
                p = scores(:, 1);
            elseif avg_scores(1) < avg_scores(2)
                subtype = 'MGMT-met';
                p = scores(:, 2);
            else
                subtype = 'Unknown'; 
                p = zeros(size(scores, 1), 1); 
            end
        otherwise
            error('Unsupported task type. Please specify "IDH" or "MGMT".');
    end

    [H, W, ~] = size(Z_for_pred);
    probMap = zeros(H, W, 'like', Z_for_pred);
    
    probMap(mask) = p;

end

function showOverlayProb(bg, prob, mask, subtype, p, mycolormap)
    bgRGB = repmat(mat2gray(double(bg)), 1, 1, 3);

    figure;
    image(bgRGB); 
    axis image off; 
    hold on;

    h = imagesc(prob);
    colormap(gca, mycolormap); 
    colorbar; 
    caxis([0 1]);

    set(h, 'AlphaData', 0.75 * double(mask));
    
    avg_prob = mean(p);
    title(sprintf('Subtype: %s  P: %.2f%%', subtype, avg_prob * 100), 'FontWeight', 'bold', 'FontSize', 16);
end
