% Function to mathematically extract the analog features using Band pass
% filter bank
%
%[1]K. M. H. Badami, S. Lauwereins, W. Meert, and M. Verhelst, “A 90 nm 
%   CMOS, 6 ?W power-proportional acoustic sensing frontend for voice 
%   activity detection,” IEEE J. Solid-State Circuits, vol. 51, no. 1, pp. 
%   291–302, Jan. 2016.
% 
% 
% Dependent functions :-
% tf()      - Generates transfer function
% lsim()    - Generates time domain response 


function [af] = an_features(audio) 
    fo1 = 0.1e3;                                    % 100-5000Hz VAD
    fo2 = 5.0e3;
    Nf = 16;                                        % 16 Band pass filters
    Gain=10;                                        % Gain of LNA (Low noise amplifier)
    fs=8000;                                        % Audio sampling frequency
    
    fc = logspace(log10(fo1), log10(fo2), Nf);      % generating exponentially spaced center frequencies
    
    % Low Noise Amplifier response common to all BPFs
    [b,a] = butter(1, 2*pi*2e3, 's');
    H_lna=Gain*tf(b,a);
    t=0:1/fs:(length(audio)-1)*1/fs;
    audio_lna = lsim(H_lna, audio, t);
    
    % Quality factor
    Q=1;
    
    % Vector to store features
    af=[];
    
    % Looping through each channel
    for i = 1:Nf
        delta_f = fc(i)/Q;                                      % Calculating bandwidth using Q
        f_3db1 = fc(i) - delta_f/2;                             % Finding 3db cutoff frequencies
        f_3db2 = fc(i) + delta_f/2;
        
        % Band pass filter response
        [b, a] = butter(1, [2*pi*f_3db1 2*pi*f_3db2], 's');     % Band pass filter definition
        H_bp = tf(b,a);                                         % Transfer function of BPF    
        y = lsim(H_bp,audio_lna,t);                             % Finding reponse to input x
        
        % Full wave rectifier
        abs_y = abs(y);                                         % Rectifying
        
        % Average finder using 16 Hz cutoff LPF
        [b,a] = butter(1, 2*pi*16, 's');                        % 16Hz low pass filter
        H_lp = tf(b,a);
        out = lsim(H_lp, abs_y, t);
        
        af(:,i) = out; 
    end
end
