clc
clear
close all

dirname = '/Users/ebotica/Library/CloudStorage/OneDrive-MassachusettsInstituteofTechnology/Desktop/MIT/Short Lab/IR Camera/Tests/20240724_041_04'; %ADJUST
filename1= dirname+string('/')+'Test_parameters.mat';
load(filename1)

%% Inputs
filenames = dir(dirname +  string('/') + 'IR_mat_raw/Calib' +  string('/*'));
filenames = filenames(3:end);
fileTable = struct2table(filenames);
sortedTable = sortrows(fileTable, 'datenum', 'ascend');
sortedFiles = sortedTable.name;
filenames = sortedFiles;

% Polynomial degree
N_poly = 2; % R_update: fitting through a F=ax+b polynomial


%% Estimating correction function coefficients
% Gathering data
for N=1:numel(filenames)
    filename2 = dirname+string('/')+ 'IR_mat_raw/Calib/'+ filenames{N,1};
    load(filename2);
    counts = COUNTS(y_tst_range,x_tst_range,:);
    avg_counts(N,1) = mean(counts(:));
    counts_2d{N,1}= mean(counts,3);
    clear COUNTS counts
    %clear counts
end


% Performing polynomial fit
for i=1:(y_tst_range(end)-y_tst_range(1)+1)
    for j=1:(x_tst_range(end)-x_tst_range(1)+1)
        % Determining polynomial coefficients and R^2 for each pixel
        for n=1:numel(filenames)
                B(n,1) = avg_counts(n,1);
                for m = 1:N_poly
                    A(n,m) = counts_2d{n,1}(i,j)^(m-1);
                end
        end
        NUC_coeff{i,j}=flipud(pinv(A)*B);
        for n=1:numel(filenames)
            counts_calc(n,1) = polyval(NUC_coeff{i,j},counts_2d{n,1}(i,j));
        end
        R2(i,j) = 1 - (sum((avg_counts-counts_calc ).^2)/sum((avg_counts-mean(counts_calc)).^2));
    end
end

% Filtering A, B Matrix
for i=1:size(NUC_coeff,1)
    for j=1:size(NUC_coeff,2)
        A(i,j) = NUC_coeff{i,j}(1);
        B(i,j) = NUC_coeff{i,j}(2);
    end
end
A = imguidedfilter(imgaussfilt(A,10));
B = imguidedfilter(imgaussfilt(B,10));
for i=1:size(NUC_coeff,1)
    for j=1:size(NUC_coeff,2)
        NUC_coeff{i,j}(1) = A(i,j);
        NUC_coeff{i,j}(2) = B(i,j);
    end
end

filename3 =  dirname+string('/')+ 'IR_mat__nuc/NUC_Coefficients.mat'  ;
save(filename3,'NUC_coeff','R2','avg_counts');


%% Plot

figure(1)
surfc(R2)
shading interp
colormap jet
saveas(figure(1),dirname+string('/')+ 'IR_mat__nuc/NUC_R2_plot.png','png')

figure(2)
surfc(A)
shading interp
colormap jet
saveas(figure(2),dirname+string('/')+ 'IR_mat__nuc/NUC_A_plot.png','png')

figure(3)
surfc(B)
shading interp
colormap jet
saveas(figure(3),dirname+string('/')+ 'IR_mat__nuc/NUC_B_plot.png','png')
