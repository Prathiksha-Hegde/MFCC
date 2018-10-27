
function feature_matrix = mfcc(y,fs,window_length, overlap_window)
        % function to calculate 12 MFCC, 1 energy feature,12 delta MFCC, 12 double delta MFCC, 1 delta energy feature , 1 double-delta energy feature for the audio signal   
        % input: 1. y : audio-signal
        % input: 2. fs : sampling frequency
        % input: 3. window_length : length of the frame required
        % input: 4. overlap_window : length of the overlap in the frames required
        % output: 1. feature_matrix : matrix of all features mentioned with dimensions number of frames x 39 features calculated 
       
        % Pass the signal through pre-empahsis filter
        % Generally, the coefficents of pre-emphasis filter is 0.97 for mfcc
        preemph = [1 -0.97];
        y= filter(1,preemph,y);

        % calculate the number of frames in the audio-signal
        % Generally, the window-length is 25ms and overlap window length is 10ms
        no_of_frames = frames_formed(y,fs,window_length,overlap_window);

        % divide the audio signal into frames of 25 ms with 10ms overlap
        frames = create_frames(y,fs,window_length,overlap_window);
        
        % calculate the energy of the signal
        energy_feature = energy_calc(no_of_frames,frames);
        
        % calculate delta energy
        first_delta_energy = delta_calc(energy_feature);

        % calculate double delta energy
        second_delta_energy = delta_calc(first_delta_energy);

        % compute dft coefficients for all the frames formed
        N =512;
        dft_vec = dft(N, no_of_frames, frames);
        
        % keep first 257 dft coefficients
        reqd_dft_coeff = dft_coeff(257, dft_vec);

        % Mel -filter bank
        lower_freq = 0;
        mel_lower_freq = Hz_to_mel(lower_freq);
        upper_freq = (fs/2);
        mel_upper_freq = Hz_to_mel(upper_freq);

        % consider 26 filters in the mel-filter bank
        no_filters= 26;
        mel_frequencies = linspace(mel_lower_freq,mel_upper_freq,no_filters+2);
        hz_freq = mel_to_Hz(mel_frequencies);
        fft_bins = zeros(length(hz_freq),1);

        % convert frequencies to fft bins
        for i =1: length(fft_bins)
            fft_bins(i) = floor((N+1)*hz_freq(i)/fs);
        end
        
        mel_filter_bank = mel_bank(512,fft_bins,26);
        % plot the mel-filter bank of 26 filters
        plot(mel_filter_bank);

        % multiply dct of each frame with all filter banks
        [rows_dft, col_dft] = size(reqd_dft_coeff);
        mel_filter = mel_filter_bank.';
        [rows_melfil, col_melfil] = size(mel_filter);
        out_fil_dft = zeros(rows_dft, rows_melfil);
        for l = 1: rows_dft
             for n =1: rows_melfil
             out_fil_dft(l,n) = dot(reqd_dft_coeff(l,:),mel_filter(n,:));
             end
        end

        % computing log of the absolute square of the dft coefficients
        log_output = log((abs(out_fil_dft).^2));

        % computing dct of the output obtained after taking log
        dct_output = zeros(size(log_output));
        [rows_dct, col_dct] = size(dct_output);
        for o =1: rows_dct
            dct_output(o,:) = (dct(log_output(o,:)));
        end

        mfcc_coeff = dct_output(:, 1:12);

        % to calculate first derivative
        first_delta_mfcc = delta_calc(mfcc_coeff);

        % to calculate second derivative
        second_delta_mfcc = delta_calc(first_delta_mfcc);

        % contenating the feature matrix
        feature_matrix = zeros(no_of_frames,39);
        feature_matrix(:,1:12)= mfcc_coeff; % 12 -mfcc features
        feature_matrix(:,13) =  energy_feature; % 1- energy feature
        feature_matrix(:,14:25) = first_delta_mfcc; % 12- delta mfcc
        feature_matrix(:,26:37) = second_delta_mfcc; % 12- double delta mfcc
        feature_matrix(:,38)= first_delta_energy; % 1-delta energy feature
        feature_matrix(:,39) = second_delta_energy; % 1- double-delta energy feature
           
end

    function energy_feature = energy_calc(no_of_frames,frames)
    % function to calculate energy of the audio signal
    % input: 1. no_of_frames : scalar value - the total number of frames created from the audio signal with given window_length and overlap_window_length
    % input: 2. frames - matrix of dimensions: no_of_frames x number of samples in each frame       
    % output: 3. energy_feature: array of energies of each frame of dimensions no_of_frames x 1     
        energy_feature =zeros(no_of_frames,1);
            for i=1:no_of_frames
                selected_frame= frames(i,:);
                energy_feature(i) = sum(selected_frame.^2);
            end
    end

    function mel_freq = Hz_to_mel (freq)
 %function to convert freq in Hz to mel frequency
 % input: 1. freq: array of all frquencies in Hz scale
 % output: 2. mel_freq: array of all frequencies in mel scale
        mel_freq = 1125 * log(1+(freq/700));
    end

    
    function hz_freq = mel_to_Hz(mel_frequencies)
    % function to convert mel frequencies to Hz
    % input: 1. mel_frequencies : array of all frequencies in mel scale
    % output: 1. hz_freq: array of all frequency bins in Hz scale       
        hz_freq= zeros(length(mel_frequencies),1);
        for i=1:length(mel_frequencies)
            hz_freq(i) = 700*(exp(mel_frequencies(i)/1125)-1);
        end
    end
    
    function dft_vec = dft (N,no_of_frames,frames)
    % function to find dft coefficents for each frame
    % input: 1. N : number of fft points
    % input: 2. no_of_frames : scalar value - the total number of frames created from the audio signal with given window_length and overlap_window_length
    % input: 3. frames : matrix of dimensions: no_of_frames x number of samples in each frame  
    % ouput: 1. dft_vec : dft coefficiens for each frame in the audio signal
        com_dft_vec = zeros(no_of_frames,N);
        dft_vec = zeros(no_of_frames,N);
        for i=1: no_of_frames
           x = frames(i,:); 
           com_dft_vec(i,:) = fft(x,N);
           dft_vec(i,:) = abs( com_dft_vec(i,:));    
        end
    end

    function reqd_dft_coeff = dft_coeff(no_of_coeff,dft_vec)
    % function to select the first few required number of coefficients of dft
    % input: 1. no_of_coeff : number of coefficients required
    % input: 2. dft_vec : output of the function that computes dft coefficients for each frame of the audio signal
    % ouput: 1. reqd_dft_coeff: matrix of dft coefficients of size number of frames x number of coefficients selected
        [rows, col] = size(dft_vec);
        reqd_dft_coeff = zeros(rows,no_of_coeff);
        for i =1: rows
            reqd_dft_coeff(i,:) = dft_vec(i,1:no_of_coeff);
        end   
    end

    function no_of_frames = frames_formed(y,fs,frame_length,overlap)
    % function to calculate number of frames of the audio signal
    % input: 1. y : audio-signal
    % input: 2. fs : sampling frequency of the audio signal
    % input: 3. frame_length : the length of the frame required.Generally it is 25ms
    % input: 4. overlap : the length of the overlap window. Generally it is 10ms
    % output: 1. no_of_frames : scalar value, number of frames created from audio signal with given window length and oerlap window length 
        samples_signal = length(y);
        number_samples_1_frame = floor(fs*frame_length);
        no_overlap_samples = floor(fs*overlap);
        no_of_frames=floor((samples_signal - number_samples_1_frame)/no_overlap_samples +1);
    end
 
  
    function frames = create_frames (y,fs,frame_length,overlap)
    % function to convert the audio signal into frames
    % input: 1. y :audio signal
    % input: 2. fs : sampling frequency
    % input: 3. frame_length : the length of the frame required, generally 25ms
    % input: 4. overlap: the length of the overlap window , generally 10ms
    % output: 1. frames : matrix of frames with dimensions number of frames x samples in each frame
        samples_signal = length(y);% number of samples in the signal
        number_samples_1_frame = floor(fs*frame_length);
        no_overlap_samples = floor(fs*overlap);
        strt_index =1;
        end_index=number_samples_1_frame;
        no_of_frames=floor((samples_signal - number_samples_1_frame)/no_overlap_samples +1);
        frames = zeros(no_of_frames,number_samples_1_frame);
        for i=1:no_of_frames
            frames(i,:)= y(strt_index:end_index);
            strt_index=strt_index+no_overlap_samples;
            end_index=strt_index+number_samples_1_frame-1;
        end 
    end
  

    % function to cal delta and double delta
    function delta_value = delta_calc(input)
    % function to calculate delta and double-delta
    % inuput: 1. input - matrix of input for which delta or double delta
    % has to be calculated, to calculate double- delta input should be delta values
    % output: 1. delta_value - matrix of delta or double delta values depending on the input to the function
        delta_matrix = zeros(size(input));
        [ rows_delta_matrix , col_delta_matrix] = size(delta_matrix);
        matrix_to_calc_delta = zeros(rows_delta_matrix+4,col_delta_matrix);
        for i =3:rows_delta_matrix+2
            matrix_to_calc_delta(i,:) = input(i-2,:);
        end
        matrix_to_calc_delta(1,:) = input(1,:);
        matrix_to_calc_delta(2,:) = input(1,:);
        matrix_to_calc_delta(end-1,:)= input(end,:);
        matrix_to_calc_delta(end,:)= input(end,:);
        [row_matrix_to_calc_delta, col_matrix_to_calc_delta] = size(matrix_to_calc_delta);

        for i = 3: row_matrix_to_calc_delta-2
            delta_matrix(i,:) = (matrix_to_calc_delta(i+1,:) - matrix_to_calc_delta(i-1,:)+2*(matrix_to_calc_delta(i+2,:) - matrix_to_calc_delta(i-2,:) ))/10;
        end

        delta_value = delta_matrix(3:rows_delta_matrix+2,:);
    end

    
    function mel_filter_bank = mel_bank (N,fft_bins,no_of_filters)
    % function to calculate mel- filter- bank
    % input: 1. N: number of fft points
    % input: 2. fft_bins: array of fft_bins calculated
    % input: 3. n: number of mel filters required
    % output: 4. mel_filter_bank: mel- filter bank with dimensions number of fft coefficients selected x number of mel filters
        mel_filter_bank = zeros(N/2+1,no_of_filters);
        for m =2: length(fft_bins) -1
            for i = fft_bins(m-1): fft_bins(m)
                 mel_filter_bank(i+1,m-1) = (i - fft_bins(m-1))/ (fft_bins(m) - fft_bins(m-1));
            end
            for i =  fft_bins(m): fft_bins(m+1)
                mel_filter_bank(i+1,m-1) = (fft_bins(m+1) - i) /  (fft_bins(m+1) - fft_bins(m));
                
            end
        end
    end