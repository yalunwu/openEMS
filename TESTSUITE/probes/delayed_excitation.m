function pass = delayed_excitation(varargin)
% Delayed step and constant E/H sources stay zero until their first update.
% See ts_options for engine selection and the common test options.
addpath(fullfile(fileparts(fileparts(mfilename('fullpath'))), 'helperscripts'));
opt = ts_options(varargin{:});
Sim_Path = ts_sim_path(mfilename('fullpath'));
pass = true;
dt = 1e-12;
delay_steps = 8;
% Zero initial fields with no active source must remain zero. This tolerance
% also accommodates the text output without imposing exact float comparisons.
zero_tol = 1e-30;

for custom = [false true]
    for type = [0 2]
        FDTD = InitFDTD('NrTS', 24, 'EndCriteria', 0, 'TimeStep', dt);
        if custom
            FDTD = SetCustomExcite(FDTD, 20e9, '1');
        else
            FDTD = SetStepExcite(FDTD);
            FDTD.ATTRIBUTE.f_max = 20e9;
        end
        FDTD = SetBoundaryCond(FDTD, repmat({'PEC'}, 1, 6));
        CSX = InitCSX();
        mesh.x = -2:2; mesh.y = -2:2; mesh.z = -2:2;
        CSX = DefineRectGrid(CSX, 1e-3, mesh);
        CSX = AddExcitation(CSX, 'source', type, [0 0 1], 'Delay', (delay_steps + 0.25)*dt);
        CSX = AddBox(CSX, 'source', 0, [-1 -1 -1], [1 1 1]);
        CSX = AddProbe(CSX, 'field', 2 + (type == 2), 'OverSampling', 100);
        CSX = AddPoint(CSX, 'field', 0, [0 0 0]);
        case_path = fullfile(Sim_Path, sprintf('custom%d_type%d', custom, type));
        mkdir(case_path);
        WriteOpenEMS(fullfile(case_path, 'model.xml'), FDTD, CSX);
        settings.LogFile = 'solver.log';
        settings.Silent = opt.Silent;
        RunOpenEMS(case_path, 'model.xml', opt.openEMS_opts, settings);
        data = load(fullfile(case_path, 'field'));
        values = data(:, 2:end);
        pass = ts_check('sample count', size(values, 1) == 25, ...
                        'custom=%d type=%d: got %d, expected 25', custom, type, size(values, 1)) && pass;
        early_peak = max(max(abs(values(1:delay_steps+1, :))));
        pass = ts_check('field before delay', early_peak < zero_tol, ...
                        'custom=%d type=%d: peak %.3g, tolerance %.3g', custom, type, early_peak, zero_tol) && pass;
        first_peak = max(abs(values(delay_steps+2, :)));
        pass = ts_check('field at delay', first_peak > zero_tol, ...
                        'custom=%d type=%d: peak %.3g, minimum %.3g', custom, type, first_peak, zero_tol) && pass;
    end
end
pass = ts_finish(opt, mfilename, pass, Sim_Path);
end
