function GPU_Setup(output)
%GPU_Setup(output) Compare CPU and strict HIP setup using identical HIP stepping.
% Requires a HIP build/device; CPU-only sweeps do not select this fixture.
% The new output directory is retained. Add the installed openEMS and CSXCAD
% Matlab interfaces to the search path before calling this function.
% Copyright (C) 2026 openEMS Project; SPDX-License-Identifier: GPL-3.0-or-later
if nargin == 0, output = tempname; end
if exist(output, 'dir'), error('GPU_Setup output directory already exists'); end
[ok, message] = mkdir(output);
if ~ok, error(message); end
for cylindrical = [false true]
    FDTD = InitFDTD('NrTS', 240, 'EndCriteria', 0, 'CoordSystem', double(cylindrical));
    FDTD = SetGaussExcite(FDTD, 3e9, 2e9);
    FDTD = SetBoundaryCond(FDTD, {'PEC' 'PEC' 'PEC' 'PEC' 'PEC' 'PEC'});
    CSX = InitCSX('CoordSystem', double(cylindrical));
    if cylindrical
        mesh.x = [0 1 2 4 6 9 12 16 20]; mesh.y = linspace(-pi, pi, 17); mesh.z = -10:2:10;
        first = [4 -0.5 -4]; last = [12 0.5 4];
        source_first = [6 0 -2]; source_last = [6 0 2]; probe = [9 0 2];
    else
        mesh.x = [-10 -8 -6 -4 -2 -1 0 1 3 5 7 10]; mesh.y = -10:2:10; mesh.z = -10:2:10;
        first = [-6 -6 -4]; last = [6 6 4];
        source_first = [0 0 -2]; source_last = [0 0 2]; probe = [3 2 2];
    end
    CSX = DefineRectGrid(CSX, 1e-3, mesh);
    CSX = AddMaterial(CSX, 'graded', 'Epsilon', 2.5, 'Kappa', 0.01);
    CSX = SetMaterialWeight(CSX, 'graded', 'Epsilon', '1/2');
    CSX = AddBox(CSX, 'graded', 1, first, last);
    CSX = AddExcitation(CSX, 'source', 0, [0 0 1]);
    CSX = AddBox(CSX, 'source', 2, source_first, source_last);
    CSX = AddProbe(CSX, 'ep', 2); CSX = AddPoint(CSX, 'ep', 0, probe);
    CSX = AddProbe(CSX, 'hp', 3); CSX = AddPoint(CSX, 'hp', 0, probe);
    results = cell(1, 2); modes = {'cpu', 'gpu'};
    for n = 1:2
        folder = fullfile(output, sprintf('cylinder_%d_%s', cylindrical, modes{n}));
        mkdir(folder); WriteOpenEMS(fullfile(folder, 'model.xml'), FDTD, CSX);
        settings.LogFile = fullfile(folder, 'openEMS.log'); settings.Silent = 1;
        RunOpenEMS(folder, 'model.xml', ['--engine=gpu --gpu-setup=' modes{n} ' --exact-endcriteria'], settings);
        results{n} = ReadUI({'ep', 'hp'}, folder);
        if n == 2
            log = fileread(settings.LogFile);
            assert(~isempty(strfind(log, 'HIP construction active')));
            assert(~isempty(strfind(log, 'HIP device construction result adopted')));
            assert(~isempty(strfind(log, 'HIP primitive bounds construction active')));
            assert(isempty(strfind(log, 'rebuilding a fresh CPU operator')));
        end
    end
    excited = false;
    for p = 1:2
        expected = results{1}.TD{p}; actual = results{2}.TD{p};
        assert(isequal(expected.t, actual.t));
        reference = [expected.val(:); expected.additional(:)]; values = [actual.val(:); actual.additional(:)];
        assert(isequal(size(reference), size(values))); assert(all(isfinite(values)));
        difference = values - reference;
        scale = max(abs(reference)); excited = excited || scale > 0;
        if scale == 0
            assert(all(difference == 0));
        else
            assert(max(abs(difference))/scale <= 1e-4);
            assert(norm(difference)/norm(reference) <= 1e-3);
        end
    end
    assert(excited);
    fprintf('PASS Octave/Matlab strict HIP setup, cylindrical=%d\n', cylindrical);
end
end
