clc
clear
close all
%% Parameters to adjust %%
dirname = '/Users/ebotica/Library/CloudStorage/OneDrive-MassachusettsInstituteofTechnology/Desktop/MIT/Short Lab/IR Camera/Tests/20240724_041_04'; %ADJUST
%memdir = '/Users/ebotica/Library/CloudStorage/OneDrive-MassachusettsInstituteofTechnology/Desktop/MIT/Short Lab/IR Camera/Tests/20240724_041_04/20240724_041_04_'; %ADJUST
memdir = '/Volumes/New Volume/Measurements/20240724_041_04_'
test_name = '20240724_041_04_'
cal_ran = 1:24
test_ran = 25:116
%%%%%%%%%%%%%%%%%%%%%%%%%%


% mkdir(dirname+string('/IR_mat_raw'))
% mkdir(dirname+string('/IR_mat_raw/Calib'))
% mkdir(dirname+string('/IR_mat_raw/Test'))
% mkdir(dirname+string('/IR_mat__nuc'))
% mkdir(dirname+string('/IR_mat__nuc/Calib'))
% mkdir(dirname+string('/IR_mat__nuc/Test'))
% mkdir(dirname+string('/IR_mat___temp'))
% mkdir(dirname+string('/IR_mat___temp/Calib'))
% mkdir(dirname+string('/IR_mat___temp/Test'))
filename1=dirname+string('/')+'Test_parameters.mat';



%% Select calibration area %%%
% Anotate x,y coordinates of the area between TC 
calfile = memdir + string('/') + test_name + string('_29/') + test_name +'_29.hcc';
[Data,Header,DerivedFromHeader,SaturatedPixels,BadPixels] = readIRCam([calfile]);
    COUNTS = reshape(Data, [], Header(1).Width, Header(1).Height);
    COUNTS = permute(COUNTS,[3 2 1]);
    COUNTS = imrotate(COUNTS,180);
reader(COUNTS(:,:,100),1,NaN,1)


%% Import Calibration files
x_cal_range_temp = [145:168]; %Enter coordinates from pop-up image (Space btw TC to average the counts)
y_cal_range_temp = [102:108]; %Enter coordinates from pop-up image (Space btw TC to average the counts)
x_tst_range = [145:168]; %Enter coordinates from pop-up image (Space where the beam will appear, should include the previous space btw TCs)
y_tst_range = [102:145]; %Enter coordinates from pop-up image (Space where the beam will appear, should include the previous space btw TCs)
save(filename1,"dirname", "memdir", "test_name", "cal_ran","test_ran", "x_cal_range_temp","y_cal_range_temp", "x_tst_range","y_tst_range")

% Extracting files names (calibration+test)
files = dir(memdir +  string('/*'));
files = files(4:end-1); % remove aditti%fileTable = struct2table(files);
aux=cell(length(files),1);
for x = 1:length(files)
    fil = strsplit(files(x).name,'_');
    fi = fil{length(strsplit(files(x).name,'_'))};
    if isempty(fi)
        fi = 0;
    else
        fi = str2double(fi);
    end
    if fi+1 > 0 && fi <= length(aux)
        aux{fi+1} = files(x).name;
    else
        warning('Index %d is out of bounds or invalid.', fi);
    end
end
clear files
temp_counts = [];


% Reading calibration temperatures from TCs and names of each test
% temperature_file = memdir + string('/')+ string('/Temperatures.txt');
temperature_file = dirname + string('/')+ string('Temperatures.txt');
fileID= fopen(temperature_file,"r" );
filen={};
while ~feof(fileID)
    line = fgetl(fileID);
    if ischar(line)
        filen{end+1, 1} = line;
    end    
end    

temperature = {}; %list with test temperatures from thermocouples
% Reading temperatures from calibration
if length(aux) == length(filen)
    % Read each line until the end of the file
    for i = 1:length(cal_ran)
        % Get the current line as a string
        t = filen(i);
        lastTwo = strsplit(t{max(1, end-1):end},'_');
        lastNums = lastTwo{2}; %just one TC for calibration
        % lastNums = (str2num(lastTwo{2})+str2num(lastTwo{3}))/2;
        temperature {end+1, 1} = lastNums;
    end
    
    % Read test names and save them into a list
    test_names = {};
    for j = test_ran
        line = filen(j);
        %line = line{1};
        % Check if the line has more than three characters
        if length(line{1}) > 6
            % Remove the first three characters
            disp(j)
            line1 = strsplit(line{1},'_');
            if line1{1} == aux{j}(18:end)
                mline = strjoin(line1(2:end));
            else
                % If the line has three or fewer characters, result is an empty string
                mline = '';
            end
            test_names {end+1, 1} = mline;    
        end
    end
    %test_names = char(test_names);
end
fclose(fileID);
%%
%Processing calibration files
for i = 1:length(cal_ran)
    files = aux(i)
    [Data,Header,DerivedFromHeader,SaturatedPixels,BadPixels] = readIRCam(memdir + string('/') + files +string('/') +files+'.hcc');
    COUNTS = reshape(Data, [], Header(1).Width, Header(1).Height);
    COUNTS = permute(COUNTS,[3 2 1]);
    COUNTS = imrotate(COUNTS,180);
    temp = temperature(i)
    filename = fullfile(dirname, 'IR_mat_raw', 'Calib', strcat(test_name, string(temp{1}), ".mat"));
    save(filename, 'COUNTS', 'Header', '-v7.3');
    reducedCOUNTS = COUNTS(x_cal_range_temp, y_cal_range_temp,:);
    ave = mean(reducedCOUNTS, "all");
    temp_counts = [temp_counts; ave];
    close all;
end
filename2=dirname+string('/')+'Temperature_counts.mat';
save(filename2,'temp_counts','temperature')

%% Import Test Files
% Matching file names with folders
for i = test_ran
    files = aux{i}
    [Data,Header,DerivedFromHeader,SaturatedPixels,BadPixels] = readIRCam(memdir + string('/') + files +string('/') +files+'.hcc');
    COUNTS = reshape(Data, [], Header(1).Width, Header(1).Height);
    COUNTS = permute(COUNTS,[3 2 1]);
    COUNTS = imrotate(COUNTS,180);
    filename = fullfile(string(dirname), 'IR_mat_raw', 'Test', strcat(strrep(filen(i),' ','_'), ".mat"));
    save(filename, 'COUNTS', 'Header', '-v7.3');
    close all;
% end
% else
%     disp('Error!: folder_names length and test_ran length do not match');
end


%% Saving test times and information

filenames = dir(memdir +  string('/*'));
filenames = filenames(3:end-1);
fileTable = struct2table(filenames);
sortedTable = sortrows(fileTable, 'datenum', 'ascend');
filename3=dirname+string('/')+'Test_times.mat';
save(filename3,'sortedTable');


