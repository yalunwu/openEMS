function pass = vulkan_fd(varargin)
% Compare CPU/GPU/automatic FD accumulation and compact TD gathering.
% Complex spectra use the solver's 0.1% field tolerance, with an absolute floor.
addpath(fullfile(fileparts(fileparts(mfilename('fullpath'))), 'helperscripts'));
opt = ts_options(varargin{:});
Sim_Path = ts_sim_path(mfilename('fullpath'));
FDTD = InitFDTD('NrTS', 129, 'EndCriteria', 0);
FDTD = SetGaussExcite(FDTD, 20e9, 10e9);
FDTD = SetBoundaryCond(FDTD, {'MUR','MUR','MUR','MUR','MUR','MUR'});
mesh.x = [-10 -7 -3 0 3 7 10]; mesh.y = -10:2:10; mesh.z = -10:2:10;
pass = [];
for mode = 0:2
    CSX = DefineRectGrid(InitCSX(), 1e-3, mesh);
    CSX = AddExcitation(CSX, 'source', 0, [0 0 1]);
    CSX = AddBox(CSX, 'source', 0, [0 0 -2], [0 0 2]);
    for field = 0:1
        name = ['fd' num2str(field)];
        CSX = AddDump(CSX, name, 'DumpType', 10+field, 'DumpMode', mode, ...
                      'FileType', 1, 'Frequency', [10e9 20e9], 'SubSampling', '2,2,2');
        CSX = AddBox(CSX, name, 0, [-10 -10 -10], [10 10 10]);
        name = ['td' num2str(field)];
        CSX = AddDump(CSX, name, 'DumpType', field, 'DumpMode', mode, ...
                      'FileType', 1, 'SubSampling', '2,2,2');
        CSX = AddBox(CSX, name, 0, [-10 -10 0], [10 10 0]);
    end
    for accumulation = {'cpu', 'gpu', 'auto'}
        selected = accumulation{1};
        folder = fullfile(Sim_Path, [num2str(mode) selected]);
        mkdir(folder); WriteOpenEMS(fullfile(folder, 'model.xml'), FDTD, CSX);
        RunOpenEMS(folder, 'model.xml', ['--engine=vulkan --numThreads=1 ' ...
                    '--vulkan-batch-size=64 --vulkan-fd=' selected ' --vulkan-td=' selected], ...
                    struct('Silent', opt.Silent));
        for field = 0:1
            [spectra{field+1}, meshes{field+1}] = ReadHDF5Dump(fullfile(folder, ['fd' num2str(field) '.h5']));
            [td{field+1}, tdmesh{field+1}] = ReadHDF5Dump(fullfile(folder, ['td' num2str(field) '.h5']));
        end
        if strcmp(selected, 'cpu')
            reference = spectra; reference_mesh = meshes;
            reference_td = td; reference_tdmesh = tdmesh;
        else
            for field = 1:2
                pass(end+1) = ts_check('FD output mesh', isequal(reference_mesh{field}.lines, meshes{field}.lines));
                pass(end+1) = ts_check('FD frequencies', isequal(reference{field}.FD.frequency, spectra{field}.FD.frequency));
                pass(end+1) = ts_check('TD output mesh', isequal(reference_tdmesh{field}.lines, tdmesh{field}.lines));
                pass(end+1) = ts_check('TD sample times', isequal(reference_td{field}.TD.time, td{field}.TD.time));
                for sample = 1:numel(reference_td{field}.TD.values)
                    x = reference_td{field}.TD.values{sample}; y = td{field}.TD.values{sample};
                    peak = max(abs(x(:))); err = max(abs(x(:)-y(:)));
                    pass(end+1) = ts_check(sprintf('TD mode=%d field=%d sample=%d', mode, field, sample), ...
                        all(isfinite(y(:))) && err<=peak*0.001+1e-18);
                end
                for frequency = 1:2
                    x = reference{field}.FD.values{frequency}; y = spectra{field}.FD.values{frequency};
                    peak = max(abs(x(:))); err = max(abs(x(:)-y(:)));
                    pass(end+1) = ts_check(sprintf('FD mode=%d field=%d frequency=%d', mode, field, frequency), ...
                        peak>0 && err<=peak*0.001+1e-18, 'relative_error=%g (tolerance 0.001)', err/peak);
                end
            end
        end
    end
end
pass = ts_finish(opt, mfilename, pass, Sim_Path);
end
