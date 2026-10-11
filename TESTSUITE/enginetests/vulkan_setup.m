function pass = vulkan_setup(varargin)
% Copyright (C) 2026 openEMS Project; SPDX-License-Identifier: GPL-3.0-or-later
% GPU construction preserves sampled fields and schedules within 0.1%.
addpath(fullfile(fileparts(fileparts(mfilename('fullpath'))), 'helperscripts'));
opt = ts_options(varargin{:});
Sim_Path = ts_sim_path(mfilename('fullpath'));
FDTD = InitFDTD('NrTS', 257, 'EndCriteria', 0);
FDTD = SetGaussExcite(FDTD, 20e9, 10e9);
FDTD = SetBoundaryCond(FDTD, repmat({'PML_3'}, 1, 6));
mesh.x = [-20 -17 -13 -9 -5 -2 0 2 5 9 13 17 20];
mesh.y = -20:4:20; mesh.z = -20:2:20;
CSX = DefineRectGrid(InitCSX(), 1e-3, mesh);
CSX = AddMaterial(CSX, 'dielectric');
CSX = SetMaterialProperty(CSX, 'dielectric', 'Epsilon', 3, 'Kappa', 0.01);
CSX = AddBox(CSX, 'dielectric', 0, [-5 -5 -5], [5 5 5]);
CSX = AddExcitation(CSX, 'source', 0, [0 0 1]);
CSX = AddBox(CSX, 'source', 1, [0 0 -2], [0 0 2]);
CSX = AddProbe(CSX, 'voltage', 0);
CSX = AddBox(CSX, 'voltage', 0, [5 0 -4], [5 0 4]);
pass = [];
selections = {'cpu', 1; 'cpu', 3; 'cpu', 0; 'gpu', 0; 'auto', 0};
for selection = 1:size(selections, 1)
    mode = selections{selection, 1}; threads = selections{selection, 2};
    label = [mode '_' num2str(threads)]; folder = fullfile(Sim_Path, label);
    mkdir(folder); WriteOpenEMS(fullfile(folder, 'model.xml'), FDTD, CSX);
    RunOpenEMS(folder, 'model.xml', ['--engine=vulkan --vulkan-setup=' mode ...
        ' --vulkan-batch-size=32 --exact-endcriteria --numThreads=' num2str(threads)], ...
        struct('Silent', 1, 'LogFile', 'solver.log'));
    values = dlmread(fullfile(folder, 'voltage'), '', 2, 0);
    logtext = fileread(fullfile(folder, 'solver.log'));
    if strcmp(mode, 'cpu')
        counts = regexp(logtext, 'CPU operator construction using (\d+) thread', 'tokens');
        selected = ~isempty(counts);
        if selected && threads>0
            selected = str2double(counts{1}{1})>=1 && str2double(counts{1}{1})<=threads;
        end
        pass(end+1) = ts_check(['CPU construction selected: ' label], selected);
    else
        pass(end+1) = ts_check('GPU construction selected', ~isempty(strfind(logtext, 'Operator constructed on GPU.')));
    end
    if selection==1
        reference = values;
    else
        pass(end+1) = ts_check('Probe schedule', isequal(reference(:,1), values(:,1)));
        err = norm(reference(:,2)-values(:,2)); scale = norm(reference(:,2));
        pass(end+1) = ts_check(['Probe values: ' label], scale>0 && all(isfinite(values(:))) && err<=0.001*scale+1e-18);
    end
end
pass = ts_finish(opt, mfilename, pass, Sim_Path);
end
