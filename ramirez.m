function [phi] = ramirez(x,fs, n, of )
    % Function to implement the Statistical VAD for labelling analog
    % features
    %
    %[1]   J. Ramirez, J. M. Gorriz, J. C. Segura, C. G. Puntonet and A. J. Rubio, "Speech/non-speech discrimination
    %       based on contextual information integrated bispectrum LRT," in IEEE Signal Processing Letters,  
    %       vol. 13, no. 8, pp. 497-500, Aug. 2006.
    %
    % Dependent functions :-
    % v_specsub()   - To find the clean version of the noisy signal by
    % spectral subtraction
    
    % Filtering the noisy signal by spectral subtraction
    x_filt = v_specsub(x,fs);
    
    phi=[];
    sgn=[];
    p_o=[];
    prob=[];
    
    
    for i=1:floor(length(x)/n)-of+1
        
            % Finding noise
            t=(i-1)*n+1:1:(i-1)*n+n*of;            
            noise = x(t)- x_filt(t);
            
            % Finding power spectrum of noisy signal using 512 pt FFT
            Snn = (abs(fft(noise,512)).^2)'/512;
            Snn0 = zeros(1,512);
            Snn0(1,1) = Snn(1);
            % Variance of the noise bispectrum coefficients
            lambda_o = 2.0*(ifft(fft(Snn,512).*fft(Snn,512)).*Snn) + 2.0*pi*var(noise)*var(noise)*Snn0;
            
            % Power spectrum of clean signal using 512 pt FFT
            Sss = (abs(fft(x_filt(t),512)).^2)'/512;
            % Variance of the noisy speech bispectrum coefficients
            lambda_1 = 2.0*(Sss+Snn).*( ifft(fft(Sss,512).*fft(Sss,512)) + ifft(fft(Snn,512).*fft(Snn,512)) + 2.0*ifft(fft(Sss,512).*fft(Snn,512)) );
            
            % Finding apriori SNR
            epsilon = (lambda_1 - lambda_o)./lambda_1;
            y=x(t).*x(t) - mean(x(t).*x(t));
            Syx = (fft(x(t),512).*conj(fft(y,512)))'/512;
            % Aposteriori SNR
            gamma = (abs(Syx).^2)./lambda_o;
            llr = (gamma.*epsilon)./(1+epsilon) - log(1+epsilon) ;
            
            % Finding mean of LLRs
            phi = [phi; mean(llr)*ones(n,1)];
    end
end
