clc
clear
close all

dirname = '/Users/ebotica/Library/CloudStorage/OneDrive-MassachusettsInstituteofTechnology/Desktop/MIT/Short Lab/IR Camera/Tests/20240724_041_04'; %ADJUST
filename1= dirname+string('/')+'Test_parameters.mat';
load(filename1)
filename0 = dirname+string('/')+ 'files_names.mat';
load(filename0);

%% Parameters to adjust %%
%Space btw TC in the new image reduced image respect previous coordinates) to calibrate average counts
x_cal_range_new = [(x_tst_range(1)-x_cal_range_temp(1)):(x_tst_range(end)-x_cal_range_temp(end))]; %Enter coordinates from pop-up image
y_cal_range_new = [(y_tst_range(1)-y_cal_range_temp(1)):(y_tst_range(end)-y_cal_range_temp(end))]; %Enter coordinates from pop-up image
x_cal_range_new = [1:24];
% y_cal_range_new = [1:31];

%% Temperature Counts correlation

filename2 = dirname+string('/')+ 'Temperature_counts.mat';
load(filename2)
t = zeros(size(temperature)); 
for i = 1:numel(temperature)
    t(i) = str2num(temperature{i});
end
temperature=t;
% filenames_new_calib = {'26_Pre_TCG38_TCP40.mat','33_Pre_TCG35_TCP37.mat','40_Pre_TCG33_TCP33.mat','47_Pre_TCG32_TCP32.mat','54_Pre_TCG30_TCP31.mat','61_Pre_TCG27_TCP28.mat','68_Pre_TCG29_TCP37.mat','75_Pre_TCG29_TCP30.mat','83_Pre_TCG28_TCP29.mat','90_Pre_TCG27_TCP27.mat','97_Pre_TCG26_TCP27.mat'};
% a=size(filenames_new_calib);
% temperature_new_calib = [40,37,33,32,31,28,37,30,29,27,27];
% 1.0 - Load Calibration files to obtain the average counts
l=1;
% for m=1:a(2)
%     % 1.1 - Importing the data
%     filename2 = dirname+string('/')+ 'IR_mat__nuc/Test/'+ filenames_new_calib{1,m};
%     load(filename2);
%     counts = COUNTS(y_cal_range_new,x_cal_range_new,:);
%     avg_counts(m,1) = mean(counts(:));
%     counts_2d{m,1}= mean(counts,3);
%     clear COUNTS counts
% end
for m=1:size(filenames{l,1},1)
    % 1.1 - Importing the data
    filename2 = dirname+string('/')+ 'IR_mat__nuc/Calib/'+ filenames{l,1}{m,1};
    load(filename2);
    counts = COUNTS(y_cal_range_new(y_cal_range_new>0),x_cal_range_new(x_cal_range_new>0),:);
    avg_counts(m,1) = mean(counts(:));
    counts_2d{m,1}= mean(counts,3);
    clear COUNTS counts
end


%% Find a fitting model for the calibration curve %%

% Define a set of candidate models
models = {'poly1', 'poly2', 'exp1', 'exp2', 'power1', 'sin1', 'fourier1'};

% Initialize variables to store the best fit and the best goodness-of-fit
bestFit = [];
bestGOF = 1e6; % Start with a large number, lower is better for residuals

% Loop over each model and fit to the data
for i = 1:length(models)
    % Try fitting the current model
    try
        n =1
        m = length(avg_counts)
        [fitresult, gof] = fit(double(avg_counts), double(temperature), models{i});

        % Check if the current fit is better based on the sum of squared errors (SSE)
        if gof.sse < bestGOF
            bestFit = fitresult;
            bestGOF = gof.sse;
        end
    catch ME
        % If fitting fails, print the error message and skip this model
        fprintf('Model %s failed with error: %s\n', models{i}, ME.message);
        continue;
    end
end
% Check if a best fit was found
if isempty(bestFit)
    error('No fitting model was successful.');
else
    cal_curve = bestFit;

    % Display the best fitting function
    disp('Best fitting function:');
    disp(bestFit);

    % Plot the original data and the best fit
    %x=1:avg_counts(end)
    figure(1);
    plot(avg_counts, temperature, 'o', 'DisplayName', 'Data');
    hold on;
    plot(avg_counts, bestFit(avg_counts), 'DisplayName', 'Best Fit');
    legend('show');
    ylabel('Measured Temperature');
    xlabel('Counts');
    title('Best Fit for the Given Data');
    grid on;
    saveas(figure(1),dirname+string('/')+ 'IR_mat__nuc/best_fit_plot.png','png')
    close(figure(1))
end

%% Check fitting with calibration images
l=1
% for m=1:size(filenames_new_calib)
%     % 1.1 - Importing the data
%     filename2 = dirname+string('/')+ 'IR_mat__nuc/Test/'+ filenames_new_calib{m,1};
%     load(filename2);
%     [Nx,Ny,Nt] = size(COUNTS);
%     for i=1:1
%         for j=1:Nx
%             for k=1:Ny
%             % 1.2 - Apply the conversion from counts to Temperature
%             COUNTS(j,k,i)=bestFit(COUNTS(j,k,i));
%             end
%         end
%     end
%     counts_temp = COUNTS(y_cal_range_new,x_cal_range_new,1);
%     avg_counts_temp(m,1) = mean(counts_temp(:));
% 
%     % 1.3 - Saving temperature images
%     filename3 = dirname + string('/')+ 'IR_mat___temp/Calib/'+ filenames_new_calib{m,1};
%     save(filename3,'COUNTS','-v7.3') 
%     clear COUNTS
% end
for m=1:size(filenames{l,1},1)
    % 1.1 - Importing the data
    filename2 = dirname+string('/')+ 'IR_mat__nuc/Calib/'+ filenames{l,1}{m,1};
    load(filename2);
    [Nx,Ny,Nt] = size(COUNTS);
    for i=1:1
        for j=1:Nx
            for k=1:Ny
            % 1.2 - Apply the conversion from counts to Temperature
            COUNTS(j,k,i)=bestFit(COUNTS(j,k,i));
            end
        end
    end
    counts_temp = COUNTS(y_cal_range_new(y_cal_range_new>0),x_cal_range_new(x_cal_range_new>0));
    avg_counts_temp(m,1) = mean(counts_temp(:));
    counts_meas = COUNTS(:,:,1);

    % 1.3 - Saving temperature images
    filename3 = dirname + string('/')+ 'IR_mat___temp/Calib/'+ filenames{l,1}{m,1};
    save(filename3,'COUNTS','counts_meas','-v7.3') 
    clear COUNTS
end
plot(avg_counts_temp, temperature, 'o', 'DisplayName', 'Data');
hold on;
plot(temperature, temperature, 'DisplayName', 'Best Fit');
legend('show');
ylabel('Measured Temperature');
xlabel('Averaged Calculated Temperature');
grid on;
saveas(figure(1),dirname+string('/')+ 'IR_mat__nuc/fit_comparison_plot.png','png')

%% applying the fitting model to Test data
l=2
for m=1:size(filenames{l,1},1)
    % 1.1 - Importing the data
    filename2 = dirname+string('/')+ 'IR_mat__nuc/Test/'+ filenames{l,1}{m,1};
    load(filename2);
    [Nx,Ny,Nt] = size(COUNTS);
    for i=1:1
        for j=1:Nx
            for k=1:Ny
            % 1.2 - Apply the conversion from counts to Temperature
            COUNTS(j,k,i)=bestFit(COUNTS(j,k,i));
            end
        end
    end
    counts_meas = COUNTS(:,:,1);


    % 1.3 - Saving temperature images
    filename3 = dirname + string('/')+ 'IR_mat___temp/Test/'+ filenames{l,1}{m,1};
    save(filename3,'COUNTS','counts_meas','-v7.3')  
    clear COUNTS counts_meas
end




