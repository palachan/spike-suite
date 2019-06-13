function [mean] = run_kilosort(fpath,bin_file,fs,n_trodes,trodetype)

    if n_trodes == 8 && strcmp(trodetype,'tetrode')
        tt8ChannelMap(strcat(fpath,'\kilofiles\'),fs);
    elseif n_trodes == 4 && strcmp(trodetype,'tetrode')
        tt4ChannelMap(strcat(fpath,'\kilofiles\'),fs);
    elseif n_trodes == 2 && strcmp(trodetype,'tetrode')
        tt2ChannelMap(strcat(fpath,'\kilofiles\'),fs);
    elseif n_trodes == 1 && strcmp(trodetype,'tetrode')
        tt1ChannelMap(strcat(fpath,'\kilofiles\'),fs);
        
    elseif n_trodes == 8 && strcmp(trodetype,'stereotrode')
        st8ChannelMap(strcat(fpath,'\kilofiles\'),fs);
    elseif n_trodes == 1 && strcmp(trodetype,'stereotrode')
        st1ChannelMap(strcat(fpath,'\kilofiles\'),fs);
    end

    master_file = 'C:\Users\Jeffrey_Taube\Desktop\Analysis\Kilosort\Patrick_run\master_script.m';

    try
        run(master_file);
    catch
        exit;
    end
    
    mean = 1;
    
    exit;
    
end