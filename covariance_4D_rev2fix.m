% covariance_4D_rev2.m
% Reading of the nmrPipe header has been updated to work for all DIMORDER
% 2018-03-30 - BH/AK/DF
function covariance_4D()

    %====================== EDIT THESE PARAMETERS ONLY =========================

    filenames = {'HCcH.pipe', 'HCcH.pipe'};
    
    extract_IJLM = {};
   
    extract_K = {};
    
    filename_4D = '4D_HCCH.pipe';
    
    labels_4D = {'H', 'C', 'Hs', 'Cs'};
    
    lambda = 0.5;
    
    with_mrs = false;
    
    downsample = [];
    
    %===========================================================================
    
%% =================== Users should not edit below here ========================
    
    % Error checking
    if size(filenames, 2) ~= 2
        msg = 'The filenames cell array should have two columns.';
        error('covar:filename', msg);
    end
    if isempty(extract_IJLM)
        extract_IJLM = repmat({[]}, 2, 2);
    elseif length(extract_IJLM) ~= 4
        msg = 'The extract_IJLM cell array should have zero or four elements';
        error('covar:extract_IJLM', msg);
    else
        extract_IJLM = reshape(extract_IJLM, 2, 2);
        for x = 1:4
            if ~isempty(extract_IJLM{x}) && length(extract_IJLM{x}) ~= 2
                msg = ['Each array in extract_IJLM should have exactly ' ...
                       'zero or two elements'];
                error('covar:extract_IJLM', msg);
            end
        end
    end
    if isempty(extract_K)
        extract_K = repmat({[]}, 1, size(filenames,1));
    elseif length(extract_K) ~= size(filenames,1)
        msg = ['The length of the extract_K cell array should be zero or ' ...
               'match the number of spectrum pairs.'];
        error('covar:extract_K', msg);
    else
        for p = 1:length(extract_K)
            if ~isempty(extract_K{p}) && length(extract_K{p}) ~= 2
                msg = ['Each array in extract_K should have exactly zero ' ...
                       'or two elements'];
                error('covar:extract_K', msg);
            end
        end
    end
    for label = labels_4D
        if length(label{:}) > 4
            msg = ['Label "' label{:} '" exceeds 4 characters in length.'];
            error('covar:labels', msg);
        end
    end
    if isempty(downsample)
        downsample = ones(1,4);
    elseif length(downsample) ~= 4
        msg = 'The downsample array should have zero or four elements.';
        error('covar:downsample', msg);
    end

    % Read spectra and headers. Extract some useful variables
    st = load_spectra( filenames, extract_IJLM, extract_K );
    I = st.I;
    J = st.J;
    L = st.L;
    M = st.M;
    np = st.num_pairs;
    for p = 1:np
        print_shape(sprintf('st.spectra{%d,1} (loaded)', p), st.spectra{p,1});
        print_shape(sprintf('st.spectra{%d,2} (loaded)', p), st.spectra{p,2});
    end

    % Prepare 4D header. Extract other useful variables
    header_4D = create_4D_header( st, downsample, labels_4D );
    I_4D = header_4D(26,1);
    J_4D = header_4D(26,2);
    L_4D = header_4D(26,3);
    M_4D = header_4D(26,4);
    print_shape('header_4D', header_4D);

    % Calculate indices to output in downsampled 4D
    IJ_write = bsxfun(@plus,(0:downsample(1):I-1)',(0:downsample(2):J-1)*I) + 1;
    LM_write = bsxfun(@plus,(0:downsample(3):L-1)',(0:downsample(4):M-1)*L) + 1;
    print_shape('IJ_write', IJ_write);
    print_shape('LM_write', LM_write);

    % Take derivative and perform first steps of GIC. Save results of SVD
    wbar = waitbar(0,'Performing SVD...');
    for p = 1:np
        % Create stacked spectrum & perform SVD
        print_shape(sprintf('pair %d: spectra{p,1} before reshape', p), st.spectra{p,1});
        print_shape(sprintf('pair %d: spectra{p,2} before reshape', p), st.spectra{p,2});
        stacked = [ reshape(st.spectra{p,1}, I*J, []); ...
                    reshape(st.spectra{p,2}, L*M, [])  ];
        print_shape(sprintf('pair %d: stacked after stacking', p), stacked);
        stacked = diff(stacked,1,2); % Take derivative
        print_shape(sprintf('pair %d: stacked after diff', p), stacked);
        size(stacked)
        [U, S, ~] = svd(stacked, 'econ');
        print_shape(sprintf('pair %d: U after svd', p), U);
        print_shape(sprintf('pair %d: S after svd', p), S);
        st.UX1_S{p} = U(IJ_write(:),:) * S ^ (2*lambda); % Only calc UX1*S once
        print_shape(sprintf('pair %d: UX1_S', p), st.UX1_S{p});
        st.UX2{p} = U(I*J + LM_write(:),:)';
        print_shape(sprintf('pair %d: UX2', p), st.UX2{p});

        % Find maxima for MRS calculations
        if with_mrs
            st.X1_max{p} = max(abs(stacked(IJ_write(:))),[],2);
            st.X2_max{p} = max(abs(stacked(I*J + LM_write(:))),[],2);
            st.X1_max{p} = reshape(st.X1_max{p}, I_4D, J_4D);
            st.X2_max{p} = reshape(st.X2_max{p}, L_4D, M_4D);
            print_shape(sprintf('pair %d: X1_max', p), st.X1_max{p});
            print_shape(sprintf('pair %d: X2_max', p), st.X2_max{p});
        end
        
        % Clear some memory
        clear U S stacked;
        st.spectra(p,:) = {[],[]};
        
        % Scale data to keep peak heights reasonable and prevent float overflow
        mps = 2^(40/np); % Max value allowed per spectrum
        bound = max(st.UX1_S{p},[],1) * max(st.UX2{p},[],2);
        print_shape(sprintf('pair %d: bound', p), bound);
        ratio = bound/mps;
        if ratio > 1
            % Scale by a power of 2 (to prevent rounding errors)
            if numel(st.UX1_S{p}) < numel(st.UX2{p})
                st.UX1_S{p} = st.UX1_S{p} ./ 2^nextpow2(ratio);
            else
                st.UX2{p} = st.UX2{p} ./ 2^nextpow2(ratio);
            end
            print_shape(sprintf('pair %d: UX1_S after scaling', p), st.UX1_S{p});
            print_shape(sprintf('pair %d: UX2 after scaling', p), st.UX2{p});
        end
        waitbar(p/np,wbar);
    end
    
    % Calculate 4D planes, take product from different pairs of spectra,
    % and write to the disk
    waitbar(0,wbar,'Writing 4D planes...');
    planes = zeros(I_4D, J_4D, np);
    print_shape('planes (initialized)', planes);
    num_percents = length(strfind(filename_4D, '%'));
    for m = 1:M_4D
        for l = 1:L_4D
            lm = sub2ind(size(LM_write), l, m);
            for p = 1:np
                print_shape('UX2{p}(:,lm)', st.UX2{p}(:,lm));
                planes(:,:,p) = reshape(st.UX1_S{p}*st.UX2{p}(:,lm),I_4D,J_4D);
                print_shape(sprintf('planes after pair %d product', p), planes);
                if with_mrs
                    ratios = st.X1_max{p} ./ st.X2_max{p}(l,m);
                    print_shape('ratios', ratios);
                    W = 1./exp(abs(log(ratios)));
                    print_shape('W', W);
                    planes(:,:,p) = planes(:,:,p) .* W;
                    print_shape(sprintf('planes after pair %d MRS weighting', p), planes);
                end
            end
            print_shape('planes before clipping negatives', planes);
            planes(planes(:) < 0) = 0;  % Anything negative is an artifact
            print_shape('planes after clipping negatives', planes);
            final_plane = prod(planes, 3);
            print_shape('final_plane', final_plane);
            if num_percents == 0
                % Single file
                if lm == 1
                    write_covar2pipe(final_plane, header_4D, filename_4D);
                else
                    fID = fopen(filename_4D, 'a');
                    fwrite(fID, final_plane, 'float32');
                    fclose(fID);
                end
            elseif num_percents == 1
                % Series of 3D cubes
                filename = sprintf(filename_4D, m);
                if l == 1
                    write_covar2pipe(final_plane, header_4D, filename);
                else
                    fID = fopen(filename, 'a');
                    fwrite(fID, final_plane, 'float32');
                    fclose(fID);
                end
            elseif num_percents == 2
                % Series of 2D planes
                filename = sprintf(filename_4D, m, l);
                write_covar2pipe(final_plane, header_4D, filename);
            end
        end
        waitbar(m/M_4D,wbar);
    end
    close(wbar);
    
    if num_percents == 0
        % FDPIPEFLAG must be set for monolithic files
        if isunix || ismac
            % Manually implement "sethdr filename_4D -pipeFlag 1"
            [~,~] = system(['printf "\x00\x00\x80\x3f" | dd of=' filename_4D ...
                            ' bs=4 count=1 seek=57 conv=notrunc']);
        elseif ispc
            wrnstr = ['You are outputting a monolithic 4D file in Windows.\n'...
                      'Please run "sethdr ' filename_4D ' -pipeFlag 1"'];
            warning(wrnstr);
        end
    end
    
    return;
end

%% Local functions

function print_shape(name, A)
    fprintf('%s: %s\n', name, mat2str(size(A)));
end

function structure = load_spectra( filenames, extract_IJLM, extract_K )
    % Read the data
    np = size(filenames, 1); % number of pairs
    headers = cell(np, 2);
    spectra = cell(np, 2);
    for p = 1:np
        headers(p,:) = {read_header(filenames{p,1}), ...
                        read_header(filenames{p,2})};
        print_shape(sprintf('load_spectra: headers{%d,1}', p), headers{p,1});
        print_shape(sprintf('load_spectra: headers{%d,2}', p), headers{p,2});
        sizes = {headers{p,1}(26,1:3), headers{p,2}(26,1:3)};
        spectra(p,:) = {read_spectrum(filenames{p,1}, sizes{1}), ...
                        read_spectrum(filenames{p,2}, sizes{2})};
        print_shape(sprintf('load_spectra: spectra{%d,1} after read', p), spectra{p,1});
        print_shape(sprintf('load_spectra: spectra{%d,2} after read', p), spectra{p,2});
    end
    
    % Determine overlapping regions
    overlap_IJLM = cell(2,2);
    for s = 1:2
        for d = 1:2
            right_max = -inf;
            left_min = inf;
            for p = 1:np
                OBS  = headers{p,s}(14,d);
                ORIG = headers{p,s}(16,d);
                SW   = headers{p,s}(20,d);
                SIZE = headers{p,s}(26,d);
                right = ORIG / OBS;
                left = (ORIG + SW*(SIZE-1)/SIZE) / OBS;
                right_max = max([right_max, right]);
                left_min = min([left, left_min]);
            end
            if left_min < right_max
                dims = reshape({'I', 'J', 'L', 'M'}, 2, 2);
                msg = ['No overlapping region found among all spectra ', ...
                       'for dimesion ', dims{d,s}];
                error('load_spectra:no_overlap', msg);
            end
            overlap_IJLM{d,s} = [left_min, right_max];
        end
    end
    overlap_K = cell(1,np);
    for p = 1:np
        right_max = -inf;
        left_min = inf;
        for s = 1:2
            OBS  = headers{p,s}(14,3);
            ORIG = headers{p,s}(16,3);
            SW   = headers{p,s}(20,3);
            SIZE = headers{p,s}(26,3);
            right = ORIG / OBS;
            left = (ORIG + SW*(SIZE-1)/SIZE) / OBS;
            right_max = max([right_max, right]);
            left_min = min([left, left_min]);
        end
        if left_min < right_max
            msg = ['No overlapping region found for dimension K in pair ' ...
                   num2str(p)];
            error('load_spectra:no_overlap', msg);
        end
        overlap_K{p} = [left_min, right_max];
    end
    
    % Apply extraction limits and update headers
    for p = 1:np
        for s = 1:2
            limits = cell(1,3);
            for d = 1:3
                OBS  = headers{p,s}(14,d);
                ORIG = headers{p,s}(16,d);
                SW   = headers{p,s}(20,d);
                SIZE = headers{p,s}(26,d);
                left_edge = (ORIG + SW*(SIZE-1)/SIZE) / OBS;
                right_edge = ORIG / OBS;
                
                % Calculate extraction limits
                if d < 3
                    left_right = sort(extract_IJLM{d,s}, 'descend');
                    if isempty(left_right)
                        left_right = overlap_IJLM{d,s};
                    end
                else
                    left_right = sort(extract_K{p}, 'descend');
                    if isempty(left_right)
                        left_right = overlap_K{p};
                    end
                end
                left_limit = left_right(1);
                right_limit = left_right(2);
                left_index = SIZE - round((OBS*left_limit - ORIG)*SIZE/SW);
                right_index = SIZE - round((OBS*right_limit - ORIG)*SIZE/SW);
                if left_index < 1
                    dims = reshape({'I', 'J', 'K', 'L', 'M', 'K'}, 3, 2);
                    msg = ['Left limit set outside spectrum in dimension ' ...
                            dims{d,s} ' of spectrum ' num2str(s) ' in pair ' ...
                            num2str(p) '\n' ...
                            'Spectrum boundary: ' num2str(left_edge) ' '...
                            'Specified limit: ' num2str(left_limit)];
                    error('load_spectra:out_of_bounds', msg);
                elseif right_index > SIZE
                    dims = reshape({'I', 'J', 'K', 'L', 'M', 'K'}, 3, 2);
                    msg = ['Right limit set outside spectrum in dimension ' ...
                            dims{d,s} ' of spectrum ' num2str(s) ' in pair ' ...
                            num2str(p) '\n' ...
                            'Spectrum boundary: ' num2str(right_edge) ' '...
                            'Specified limit: ' num2str(right_limit)];
                    error('load_spectra:out_of_bounds', msg);
                end
                limits{d} = left_index:right_index;
                
                % Update the headers
                NEW_X1 = left_index;
                NEW_XN = right_index;
                NEW_ORIG = ORIG + SW/SIZE * (SIZE - NEW_XN);
                NEW_SIZE = NEW_XN - NEW_X1 + 1;
                NEW_SW   = SW/SIZE * NEW_SIZE;
                NEW_DATA = [NEW_ORIG NEW_SW NEW_X1 NEW_XN NEW_SIZE]';
                headers{p,s}([16 20 23 24 26],d) = NEW_DATA;
            end
            % Extract
            print_shape(sprintf('load_spectra: spectra{%d,%d} before extraction', p, s), spectra{p,s});
            spectra{p,s} = spectra{p,s}(limits{:});
            print_shape(sprintf('load_spectra: spectra{%d,%d} after extraction', p, s), spectra{p,s});
        end  
    end
    
    % Verify compatible sizes between spectra
    IJLM = cell(2,2);
    for s = 1:2
        for d = 1:2
            sizes = zeros(1, np);
            for p = 1:np
            	sizes(p) = headers{p,s}(26,d);
            end
            if any(sizes ~= sizes(1))
                dims = reshape({'I', 'J', 'L', 'M'}, 2, 2);
                msg = ['Resolution mismatch in dimension ' dims{d,s}];
                error('load_spectra:mismatch', msg);
            end
            IJLM{d,s} = sizes(1);
        end
    end
    for p = 1:np
        if headers{p,1}(26,3) ~= headers{p,2}(26,3)
            msg = ['Resolution mismatch in dimension K of pair ' num2str(p)];
            error('load_spectra:mismatch', msg);
        end
    end
    
    % Build the structure
    structure = struct();
    structure.num_pairs = np;
    structure.headers = headers;
    structure.spectra = spectra;
    [structure.I, structure.J, structure.L, structure.M] = IJLM{:};
    
    return;
end

function header = read_header(filename)
    % Read raw header data
    if isempty(strfind(filename, '%'))
        fileID = fopen(filename,'r');
    else
        fn = sprintf(filename, 1);
        fileID = fopen(fn,'r');
    end
    raw_header = fread(fileID, 512, 'float32');
    print_shape('read_header: raw_header', raw_header);

    % Create covariance toolbox style headers
    header = get_header_data(raw_header);
    print_shape('read_header: header after get_header_data', header);

    % In nmrPipe headers, only the SIZE variables are rearranged when
    % transposing. Use the FDDIMORDER variables to rearrange the rest
    % of the header to match the data.
    header(1:25,[2 1]) = header(1:25,1:2);
    %header(1:25,raw_header(25:28)) = header(1:25,:);
    header(1:25,:) = header(1:25,raw_header(25:28));
    
    % Change any instance of a dimension size = 0 to dimension size = 1
    header(26,:) = max(header(26,:), 1);
    print_shape('read_header: header final', header);

    return;
end

function spec = read_spectrum(filename, read_size)
    % Read a spectrum. Can accept monolithic or plane-by-plane format
    if isempty(strfind(filename, '%'))
        fileID = fopen(filename,'r');
        fseek(fileID, 2048, 'bof');
        raw = fread(fileID,prod(read_size),'float32');
        print_shape('read_spectrum: raw data before reshape', raw);
        spec = reshape(raw, read_size);
        print_shape('read_spectrum: spec after reshape', spec);
        fclose(fileID);
    else
        spec = zeros(read_size);
        for j = 1:read_size(3)
            fn = sprintf(filename, j);
            fileID = fopen(fn, 'r');
            fseek(fileID, 2048, 'bof');
            spec(:,:,j) = fread(fileID, read_size(1:2), 'float32');
            fclose(fileID);
        end
        print_shape('read_spectrum: spec after plane-by-plane read', spec);
    end
    return;
end

function header = create_4D_header( structure, downsample, labels )
    % Use the headers of the first spectrum pair as a template
    header = [ structure.headers{1,1}(:,1) structure.headers{1,1}(:,2) ...
               structure.headers{1,2}(:,1) structure.headers{1,2}(:,2) ];
    print_shape('create_4D_header: header before downsample', header);
    % Adjust the headers for downsampling
    for d = 1:4
        DS = downsample(d);
        ORIG = header(16,d);
        SW   = header(20,d);
        SIZE = header(26,d);
        XN   = header(24,d);
        NEW_XN   = XN - mod(SIZE-1, DS);
        NEW_ORIG = ORIG + SW/SIZE * (XN - NEW_XN);
        NEW_SIZE = ceil(SIZE/DS);
        NEW_SW   = SW/SIZE * NEW_SIZE * DS;
        header([16 20 24 26],d) = [NEW_ORIG NEW_SW NEW_XN NEW_SIZE]';
    end
    % Write the new dimension labels
    header(12,:) = cellfun( @(x) typecast([uint8(x) zeros(1, 4-length(x))], ...
                                          'single'), labels);
    print_shape('create_4D_header: header final', header);
    return;
end
