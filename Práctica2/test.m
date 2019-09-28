files = dir('./Dataset/Train/*mask.png');
for i = 1:length(files)
   full_name = strcat(files(i).folder,'\',files(i).name);
   img =  imread(full_name);
   save(strcat(full_name(1:end-4), '.mat'), 'img');
end

files = dir('./Dataset/Test/*mask.png');
for i = 1:length(files)
   full_name = strcat(files(i).folder,'\',files(i).name);
   img =  imread(full_name);
   save(strcat(full_name(1:end-4), '.mat'), 'img');
end


