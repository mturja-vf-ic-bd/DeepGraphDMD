% Keisuke Fujii
 

function f = batch_wise_DeepgDMD(s)
    % You need to download Tensor Train (TT) Toolbox :
    % https://github.com/oseledets/TT-Toolbox
    % Please insert the path of the folder below:
    path_TT = 'TT-Toolbox-master';
    addpath(genpath(path_TT));
    trial = 0;
    
    % Load learned matrices
    % load('kura.mat');
    % size(kura_mat)
    % load('A_train_w_64_single_hcp.mat');
    DATAPATH = "/pine/scr/m/t/mturja/HCP_PTN1200/latent_mat/3T_HCP1200_MSMAll_d50_ts2/";
    load(DATAPATH + '/' + s + '_trial=' + num2str(trial) + '.mat');
    size(X)
    
    % Perform GraphDMD
    window_size = 64;
    feat_dim = 16;
    step = 1;
    th = 0.15;
    % output_fname = "TaskfMRI_" + num2str(window_size) + '_' + num2str(th);
    folder_name = "rsfMRIdgdmd_" + num2str(window_size) + '_' + num2str(feat_dim) + '_' + num2str(th) + '_trial=' + num2str(trial) + '/' + s;
    N = size(X, 2);
    mkdir(folder_name);
    for sw = 1:(size(X, 1) - window_size)
        x = X(sw: sw + window_size - 1, :, :);
        tx = zeros(N, N, window_size - 1);
        ty = zeros(N, N, window_size - 1);
        for i = 1:window_size - 1
            tx(:, :, i) = corrcoef(transpose(squeeze(x(i, :, :)))) - eye(N);
            ty(:, :, i) = corrcoef(transpose(squeeze(x(i + 1, :, :)))) - eye(N);
        end
        dt = 1;
        ttX = tt_tensor(tx);
        ttY = tt_tensor(ty);
        [Phi, Lambda, Omega, Psi] = tt_dmd(ttX, ttY, dt);
        Lambda = diag(Lambda);
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
    end
    disp(['TDMD ', s, ' is done'])
    clear; close all
end

