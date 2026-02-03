clc
clear
close all

dirname = '/Users/ebotica/Library/CloudStorage/OneDrive-MassachusettsInstituteofTechnology/Desktop/MIT/Short Lab/IR Camera/Tests/20240724_041_04'; %ADJUST
filename1= dirname+string('/')+'Test_parameters.mat';
load(filename1)

%% 0 - Inputs

% 0.1 -  Find and sort all .mat files
% 0.1.1 - Calibration
filenames_calib = dir(dirname +  string('/') + 'IR_mat_raw/Calib' +  string('/*'));
filenames_calib = filenames_calib(3:end);
fileTable_calib = struct2table(filenames_calib);
sortedTable = sortrows(fileTable_calib, 'datenum', 'ascend');
sortedFiles = sortedTable.name;
filenames{1,1} = sortedFiles;
clear filenames_calib

% 0.1.2 - Tests

aux = dir(dirname +  string('/') + 'IR_mat_raw/Test' +  string('/*'));
aux = aux(3:end);
aux = struct2table(aux);
sortedTable_aux = sortrows(aux, 'datenum', 'ascend');
sortedFiles_aux = sortedTable_aux.name;
filenames_aux = sortedFiles_aux;
filenames{2,1} = filenames_aux;

filename0 = dirname+string('/')+ 'files_names.mat';
save(filename0,'filenames');

%% Performing NUC
% 1.0 - Load NUC Coefficients
filename1 = dirname+string('/')+ 'IR_mat__nuc/NUC_Coefficients.mat';
load(filename1);
for l=1:size(filenames,1)
    l
    for m=1:size(filenames{l,1},1)
        m
        % 1.1 - Importing the data
        if l == 1
            filename2 = dirname+string('/')+ 'IR_mat_raw/Calib/'+ filenames{l,1}{m,1};
            load(filename2);
        else
            filename3 = dirname+string('/')+ 'IR_mat_raw/Test/'+ filenames{l,1}{m,1};
            load(filename3);
        end 

        x1 = x_tst_range(1);
        x2 = x_tst_range(end);
        y1 = y_tst_range(1);
        y2 = y_tst_range(end);
        COUNTS = single(COUNTS(y1:y2,x1:x2,:)); 
        [Nx,Ny,Nt] = size(COUNTS);

        % 1.2 - Applying NUC 
        for i=1:Nt
            for j=1:Nx
                for k=1:Ny
                COUNTS(j,k,i)=NUC_coeff{j,k}(1,1)*COUNTS(j,k,i)+NUC_coeff{j,k}(2,1);
                end
            end
        end
        
       
        %   %1.4 - Filtering 

        % 1.4.1 - Temporal filter
        for j=1:Nx
            for k=1:Ny
                COUNTS_filtered(j,k,:) = smooth(COUNTS(j,k,:),'lowess');
            end
        end
        % 1.4.2 - Spatial filter
        COUNTS_filtered = COUNTS;
        for j=1:Nt
            COUNTS_filtered(:,:,j) = imguidedfilter(COUNTS_filtered(:,:,j));
            %COUNTS_filtered(:,:,j) = wiener2(squeeze(COUNTS_filtered(:,:,j)));
        end

%         maxmin_values = [];
%         for i=1:size(COUNTS,3)
%             maxmin_values(i,1) = max(max(max(COUNTS(:,:,i))));
%             maxmin_values(i,2) = min(min(min(COUNTS(:,:,i))));
%         end
%         max_q = prctile(maxmin_values(:,1),99);
%         min_q = max([0 prctile(maxmin_values(:,2),1)]);
%         COUNTS_gray = mat2gray(COUNTS,double([min_q max_q]));
%         COUNTS_gray = single(COUNTS_gray);
%         % 2 - 3D Coherence filter the gray q" matrix 
%         Options.Scheme = 'S';
%         Options.T = 1*0.25;
%         Options.alpha = 4*0.001;
%         Options.eigenmode = 4;
%         block_size = 100;
%         residue = size(COUNTS,3);
%         i = 0;
%         while residue > 1
%             index_start = 1 + i*block_size;
%             if residue < block_size; index_end = size(COUNTS,3); residue = 0; else; index_end = index_start-1+block_size; residue = residue + (index_start-index_end-1); end
%             index_start
%             index_end
%             COUNTS_gray(:,:,index_start:index_end) = CoherenceFilter(COUNTS_gray(:,:,index_start:index_end),Options);
%             residue
%             i = i+1;
%         end
%         COUNTS_filtered = min_q+(max_q-min_q).*COUNTS_gray;
%         clear COUNTS_gray

        % 4 - Plot a comparison between original and gray filtered images
%         figure
%         subplot(1,2,1)
%         imagesc(COUNTS(:,:,300))
%         colormap jet
%         axis square
%         axis off
%         title('Original Image')
%         subplot(1,2,2)
%         imagesc(COUNTS_filtered(:,:,300))
%         colormap jet
%         axis square
%         axis off
%         title('Filtered Image')
%         figure
%         imagesc(COUNTS_filtered(:,:,300)-COUNTS(:,:,300))
%         drawnow
% 
%         COUNTS = COUNTS_filtered;
%         clear COUNTS_filtered

        % 1.5 - Saving the data    
        if l == 1
            filename4 = dirname + string('/')+ 'IR_mat__nuc/Calib/'+ filenames{l,1}{m,1};
            save(filename4,'COUNTS','-v7.3')
        else
            filename5 = dirname + string('/')+ 'IR_mat__nuc/Test/'+ filenames{l,1}{m,1};
            save(filename5,'COUNTS','-v7.3')
        end  
        clear COUNTS COUNTS_filtered
    end
end

