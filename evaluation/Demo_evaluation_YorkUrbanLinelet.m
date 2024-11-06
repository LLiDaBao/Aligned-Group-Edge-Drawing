clear; close all

dir_db = './DB/YorkUrbanDB';
addpath(genpath('./toolbox/'));
addpath('./funcs/')
load([dir_db '/our_annotation/Image_ID_List.mat']); % We get Image_ID_List
num_im = size(Image_ID_List, 1);

name_method = {'\fontname{times}Linelet'};
method2test = {'linelet'};
num_method = length(method2test);

dir_method = {'proposed'};
var_method = {'line_own'};

stats(num_method,1) = struct('prec', [], 'rec', [], 'iou', []);

bSaveResult = false;

% True positive conditions
eval_param.thres_dist = 1;
eval_param.thres_ang = pi*5/180;
eval_param.thres_length_ratio = .75; % .75 (Con1) or .5 (Con2)

for i_im = 1 : num_im
    try
        str_im = sprintf('%s/%s/%s.jpg', dir_db, Image_ID_List(i_im).name, Image_ID_List(i_im).name);
        im = imread(str_im);
        imw = im; imw(:) = 255;
        im_gray = rgb2gray(im);
        size_im = size(im_gray);
        
        % load line gnd
        str_gnd = sprintf('%s/our_annotation/%s_GND.mat', dir_db, Image_ID_List(i_im).name);
        
        if ~exist(str_gnd, 'file'), continue; end
        load(str_gnd); % we get line_gnd
        line_gnd = unique(line_gnd, 'rows');
        % Remove invalid ground truth
        idx_invalid = find( line_gnd(:,1) == 0 & line_gnd(:,2) == 0 & line_gnd(:,3) == 0 & line_gnd(:,4) == 0 );
        line_gnd(idx_invalid,:) = [];
        
        % Rearrange line segment so that elements become (x1, y1, x2, y2, center_x, center_y, length, angle)
        cp = [line_gnd(:,1) + line_gnd(:,3) line_gnd(:,2) + line_gnd(:,4)]/2;
        dx = line_gnd(:,3) - line_gnd(:,1); dy = line_gnd(:,4) - line_gnd(:,2);
        line_gnd = [line_gnd, cp, sqrt(dx.^2 + dy.^2), atan2(dy, dx)];
        
        for k = 1:num_method
            % Load estimation results
            str_est = sprintf('result/my/%s.csv', Image_ID_List(i_im).name);
            if ~exist(str_est, 'file'), continue; end
            segments = load(str_est);
            
            center = 0.5.*segments(:,1:2) + 0.5.*segments(:,3:4);
            dx = segments(:,3) - segments(:,1);
            dy = segments(:,4) - segments(:,2);
            
            length = sqrt(dx .* dx + dy .* dy);
            angle = atan2(dy, dx);
            
            line_est = [segments center length angle];
            
            % Evaluate
            if size(line_est,1) > 0
                [pr, re, iou] = evaluate_line_segment(line_est, line_gnd, eval_param);
                
                stats(k).prec(i_im,:) = pr;
                stats(k).rec(i_im,:) = re;
                stats(k).iou(i_im,:) = iou;
            else
                stats(k).prec(i_im,:) = 0;
                stats(k).rec(i_im,:) = 0;
                stats(k).iou(i_im,:) = 0;
            end
        end
        
    catch err
        fprintf('error at i_im: %d.\n', i_im);
        rethrow(err);
    end
end

%% Display scores
AP = zeros(num_method, 1);
AR = zeros(num_method, 1);
IOU = zeros(num_method, 1);

for k = 1:num_method
    AP(k) = mean(stats(k).prec,1);
    AR(k) = mean(stats(k).rec,1);
    IOU(k) = mean(stats(k).iou,1);
end
F_sc = 2 * (AP .* AR) ./ (AP + AR);

fig = figure(1); clf;
axes1 = axes('Parent',fig,'Layer','top','FontWeight','bold','FontSize',12,...
    'FontName','Times New Roman', 'XTick', 1:num_method, 'XTickLabel',name_method);
box(axes1,'on');    hold(axes1,'on');
title('Average precision and recall')
bar([AP, AR, IOU, F_sc])
hleg = legend('AP', 'AR', 'IOU', 'F-score', 'Location', 'nw');

fprintf('Final scores...\n');
[AP, AR, IOU, F_sc]