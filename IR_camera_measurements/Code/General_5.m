clc
clear
close all

%% Parameters to adjust %%
dirname = '/Users/ebotica/Library/CloudStorage/OneDrive-MassachusettsInstituteofTechnology/Desktop/MIT/Short Lab/IR Camera/Tests/20240724_041_04'; %ADJUST
filename0 = dirname+string('/')+ 'files_names.mat';
load(filename0);
filename00 = dirname+string('/')+ 'Test_times.mat';
load(filename00);

%% Visualize affected area by irradiation
% Choose a file with high ion beam power
filename1=dirname+string('/')+ 'IR_mat___temp/Test/'+ filenames{2}{44};
load(filename1)
reader(counts_meas(:,:,:),1,NaN,1)



%% Calculate average temperature of all those images in that area
%Enter beam spot area to average counts and obtain the average temperature
%of the beamspot
y_beam_range = [29:43];
x_beam_range = [7:23];

n=0;
for l=1:size(filenames,1)
    l
    for m=1:size(filenames{l,1},1)
        m
        n=n+1;
        info_sum(n,1) = sortedTable.(1)(n);
        info_sum(n,3) = sortedTable.(3)(n);
        % 1.1 - Importing the data
        if l == 1
            filename2 = dirname+string('/')+ 'IR_mat___temp/Calib/'+ filenames{l,1}{m,1};
            load(filename2);
            counts = COUNTS(y_beam_range,x_beam_range,:);
            counts_t = counts_meas(y_beam_range,x_beam_range,:);
            info_sum(n,4) = num2cell(mean(counts(:)));
            info_sum(n,5) = num2cell(mean(counts_t(:)));
            info_sum(n,2) = cellstr(filenames{l,1}{m,1});
            % clear COUNTS counts counts_meas counts_t
        else
            filename3 = dirname+string('/')+ 'IR_mat___temp/Test/'+ filenames{l,1}{m,1};
            load(filename3);
            counts = COUNTS(y_beam_range,x_beam_range,:);
            counts_t = counts_meas(y_beam_range,x_beam_range,:);
            info_sum(n,4) = num2cell(mean(counts(:)));
            info_sum(n,5) = num2cell(mean(counts_t(:)));
            info_sum(n,2) = cellstr(filenames{l,1}{m,1});
            % clear COUNTS counts counts_meas counts_t
        end 
    end
end
% filename4 = dirname+string('/')+ 'Test_summary.mat';
% save(filename4,'info_sum');

