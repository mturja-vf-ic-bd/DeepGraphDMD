% Keisuke Fujii
 

function f = batch_wise_gDMD_global(b)
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
    X = X(:, 1:1200, :);
    X = permute(X, [1 3 2]);
    size(X)
    
    % Perform GraphDMD
    feat_dim = 16;
    th = 0.15;
    output_fname = "rsfMRI_global_" + num2str(feat_dim) + '_' + num2str(th);
    N = size(X, 2);
    
    folder_name = strcat(output_fname + '/b_', num2str(b));
    mkdir(folder_name);
    x = squeeze(X(b, :, :));
    tx = zeros(N, N, size(x, 2) - feat_dim);
    ty = zeros(N, N, size(x, 2) - feat_dim);
    for i = 1:size(x, 2) - feat_dim
        tx(:, :, i) = corrcoef(transpose(x(:, i:i+feat_dim-1))) - eye(N);
        ty(:, :, i) = corrcoef(transpose(x(:, i+1:i+feat_dim))) - eye(N);
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
    fname = strcat(folder_name, '/sw_1.mat');
    save(fname, "Phi", "tx", "Psi", "Lambda", "Omega", "b0");
    disp(['TDMD ', num2str(b), ' is done'])
    clear; close all
end

