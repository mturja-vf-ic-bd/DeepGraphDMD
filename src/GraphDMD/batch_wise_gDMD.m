% Keisuke Fujii
 

function f = batch_wise_gDMD(b)
    % You need to download Tensor Train (TT) Toolbox :
    % https://github.com/oseledets/TT-Toolbox
    % Please insert the path of the folder below:
    path_TT = 'TT-Toolbox-master';
    addpath(genpath(path_TT));
    
    % Load learned matrices
    % load('kura.mat');
    % size(kura_mat)
    % load('A_train_w_64_single_hcp.mat');
    load("rsfMRI.mat");
    trial = 3;
    if trial == 1
        disp("Trial 1 ...")
        X = X(:, 1:1200, :);
    elseif trial == 2
        disp("Trial 2 ...")
        X = X(:, 1200:2400, :);
    elseif trial == 3
        disp("Trial 3 ...")
        X = X(:, 2400:3600, :);
    elseif trial == 4
        disp("Trial 4 ...")
        X = X(:, 3600:4800, :);
    end
%     X = X(:, 1:1200, :);
    X = permute(X, [1 3 2]);
%     load("trial_0.mat")
    b
%     load("trial_0.mat")
%     b = str2num(b)
    
    % Perform GraphDMD
    window_size = 64;
    feat_dim = 8;
    step = 1;
    th = 0.12;

    % output_fname = "TaskfMRI_" + num2str(window_size) + '_' + num2str(th);
    output_fname = "rsfMRIfull_window=" + num2str(window_size) + '_featdim=' + ...
        num2str(feat_dim) + '_th=' + num2str(th) + '_step=' + ...
        num2str(step) + '_trial=' + num2str(trial);

    N = size(X, 2);
    folder_name = strcat(output_fname + '/b_', num2str(b));
    mkdir(folder_name);
    for sw = 1:floor((size(X, 3) - feat_dim - window_size) / step)
        start_ind = (sw - 1) * step + 1;
        end_ind = (sw - 1) * step + 1 + window_size;
%         disp(['(', num2str(start_ind), ', ', num2str(end_ind), ')'])
        x = squeeze(X(b, :, start_ind: end_ind + feat_dim));
        tx = zeros(N, N, window_size - 1);
        ty = zeros(N, N, window_size - 1);
        for i = 1:size(x, 2) - feat_dim
            tx(:, :, i) = corrcoef(transpose(x(:, i:i+feat_dim-1))) - eye(N);
            ty(:, :, i) = corrcoef(transpose(x(:, i+1:i+feat_dim))) - eye(N);
        end
        dt = 1;
        ttX = tt_tensor(tx);
        ttY = tt_tensor(ty);
        [Phi, Lambda, Omega, Psi] = tt_dmd(ttX, ttY, dt);
        Lambda = diag(Lambda);
    %         Phi = Phi(:, :, 1:r);
    %         Lambda = Lambda(1:r);
    %         Omega = Omega(1:r);
    %         Psi = Psi(1:r);
        % Compute initial value
        A0 = reshape(tx(:, :, 1), N * N, 1);
        Phi_inv = pinv(reshape(Phi, N * N, size(Phi, 3)));
        b0 = Phi_inv * A0;

        Phi = Phi(:, :, Psi < th);
        Omega = Omega(Psi < th);
        Lambda = Lambda(Psi < th);
        Psi = Psi(Psi < th);
        b0 = b0(Psi < th);
        fname = strcat(folder_name, '/sw_', num2str(sw),'.mat');
        save(fname, "Phi", "Psi", "Lambda", "Omega", "b0");
        disp(['TDMD ', num2str(b), ' ', num2str(sw), ' is done'])
    end
    disp("Completed!")
    clear; close all
end

