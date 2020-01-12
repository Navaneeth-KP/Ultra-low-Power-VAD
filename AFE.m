 
% CODE TO EXTRACT ANALOG FEATURES FOR TRAINING THE BINARIZED NEURAL NETWORK CLASSIFIER

% References - 
% [1]   J. Ramirez, J. M. Gorriz, J. C. Segura, C. G. Puntonet and A. J. Rubio, "Speech/non-speech discrimination
%       based on contextual information integrated bispectrum LRT," in IEEE Signal Processing Letters,
%       vol. 13, no. 8, pp. 497-500, Aug. 2006.
%  Statistical VAD parameters - Frame length - 160ms, Frame shift - 80ms (Overlap factor - 2)


% Dependent functions:-
% an_features() - Function to extract analog features
% ramirez()     - Function implementing Statistical VAD using Integrated bispectrum
% roc_ramirez() - Function to generate the ROC curve for Statistical VAD

    % noise parameters
    noise_type = 'babble';
    snr='5';

    % Threshold for LRT (Likelihood Ratio Test) for Statistical VAD[1]
    llr_threshold = 0;

    A=[];
    B=[];

    % GENERATING TRAINING DATA

    for i=1:24
        disp(i);

        if i<10
            File = strcat('F:\ProgramFiles\MATLAB\NOIZEUS\',noise_type,'_',snr,'dB\',snr,'dB\','sp0',string(i),'_',noise_type,'_sn',snr,'.wav');
            File_clean =strcat('F:\ProgramFiles\MATLAB\NOIZEUS\clean\sp0',string(i),'.wav');
        else 
            File = strcat('F:\ProgramFiles\MATLAB\NOIZEUS\',noise_type,'_',snr,'dB\',snr,'dB\','sp',string(i),'_',noise_type,'_sn',snr,'.wav');
            File_clean =strcat('F:\ProgramFiles\MATLAB\NOIZEUS\clean\sp',string(i),'.wav');
        end

        % Reading audio
        [a,fs] = audioread(File);
        [b,fs] = audioread(File_clean);

        % Appending audio into a single array
        A=[A; a];
        B=[B;b];

    end
    % Extracting analog features for clean speech and only noise
    N=A-B;    % finding noise
    [af_speech] = an_features(A);
    [af_noise] = an_features(N);
    
    % Running Statistical VAD on clean speech to generate Labels.
    [LLR] = ramirez(B,fs, 80, 2);
    label_speech = (LLR > llr_threshold);
    % labeling analog features for only noise to zero
    label_noise = zeros(length(af_noise),1);
    
    % Ignoring first 80 samples since 2 frames required by the Statistical
    % VAD.
    af_new_speech = af_speech(81:length(label_speech),:);
    % Taking enough noise samples equal to that of speech samples
    af_new_noise = af_noise(81:-length(label_speech)+2*sum(label_speech),:);
    
    L_speech= label_speech(1:end-80);
    L_noise=label_noise(1:length(af_new_noise));
    
    % Oversampling training data to ensure higher TPR (True positive rates)
    data1 = [[af_new_speech L_speech];
        [af_new_speech L_speech];[af_new_noise(1:end,:)  L_noise(1:end)]];

    A=[];
    B=[];

    LLR_clean=[];
    LLR_n=[];

    % GENERATING TESTING DATA
    for i=25:30
        disp(i);

        if i<10
            File = strcat('F:\ProgramFiles\MATLAB\NOIZEUS\',noise_type,'_',snr,'dB\',snr,'dB\','sp0',string(i),'_',noise_type,'_sn',snr,'.wav');
            File_clean =strcat('F:\ProgramFiles\MATLAB\NOIZEUS\clean\sp0',string(i),'.wav');
        else 
            File = strcat('F:\ProgramFiles\MATLAB\NOIZEUS\',noise_type,'_',snr,'dB\',snr,'dB\','sp',string(i),'_',noise_type,'_sn',snr,'.wav');
            File_clean =strcat('F:\ProgramFiles\MATLAB\NOIZEUS\clean\sp',string(i),'.wav');
        end

        [a,fs] = audioread(File);
        [b,fs] = audioread(File_clean);
        
        % Running Statistical VAD for each file for generating ROC curve
        LLR_clean = [LLR_clean; ramirez(b,fs,80,2)];
        LLR_n = [LLR_n; ramirez(a,fs,80,2)];
        
        % Appending all samples into single array
        A=[A; a];
        B=[B;b];

    end

        % Extracting analog features
        N=A-B;   % Finding noise
        [af_speech] = an_features(A);
        [af_noise] = an_features(N);
        
            % Running Statistical VAD on clean speech to generate Labels.
        [LLR] = ramirez(B,fs, 80, 2);
        label_speech = (LLR > llr_threshold);
        % labeling analog features for only noise to zero
        label_noise = zeros(length(af_noise),1);

        % Ignoring first 80 samples since 2 frames required by the Statistical
        % VAD.
        af_new_speech = af_speech(81:length(label_speech),:);
        % Taking enough noise samples equal to that of speech samples
        af_new_noise = af_noise(81:-length(label_speech)+2*sum(label_speech),:);

        L_speech= label_speech(1:end-80);
        L_noise=label_noise(1:length(af_new_noise));
        
        % Complete test data with equal speech,non-speech samples
        data2 = [[af_new_speech L_speech];[af_new_noise  L_noise]];

        audio_noisy = A;
        audio_clean = B;
        
        %roc plot
        T1= -10:0.1:1000;   
        [x1,y1] = roc_ramirez(LLR_clean, LLR_n, T1 ,llr_threshold);
        figure(1)
        plot(x1,y1,'b--');
        hold on
        xlim([0 1])
        ylim([0 1])
        grid on

        % Saving it into .mat file required for training
        save(strcat('train_speech_',snr,'_',noise_type,'.mat'),'data1','data2','audio_noisy','audio_clean','x1','y1');
