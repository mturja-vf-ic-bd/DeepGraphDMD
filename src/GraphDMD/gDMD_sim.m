function f = gDMD_sim()
    path_TT = 'TT-Toolbox-master';
    addpath(genpath(path_TT));
    load("sim_data.mat");
    ttX = tt_tensor(data(:, :, 1:end-1));
    ttY = tt_tensor(data(:, :, 2:end));
    dt=1;
    [Phi, Lambda, Omega, Psi] = tt_dmd(ttX, ttY, dt);
    save("gDMD_sim_results.mat", "Phi", "Psi", "Lambda", "Omega");