function [TN,TP] = roc_ramirez(decision_actual, decision_predict, threshold,t)
    % Function to find the true postive rate and true negative rate for an
    % array of threshold inputs, generating the roc curve. 
    
    TN =[];
    TP =[];
    decision_threshold = zeros(length(decision_actual),1);
    decision_actual_t = zeros(length(decision_actual),1);
    
    for i =1:length(threshold) 
        for k =1:length(decision_predict)
            if decision_predict(k) > threshold(i)
                decision_threshold(k) = 1;
            else
                decision_threshold(k) = 0;
            end
            
            if decision_actual(k) > t
                decision_actual_t(k) = 1;
            else
                decision_actual_t(k) = 0;
            end
            
        end
        
        ones = sum(decision_actual_t);
        zero = length(decision_actual_t) - ones;
        count_ones=0;
        count_zeros=0;
        
        for j = 1:length(decision_actual)
            if decision_actual_t(j) == decision_threshold(j)
                if decision_actual_t(j)==1
                    count_ones =  count_ones+1;
                else 
                    count_zeros = count_zeros + 1;
                end
            end
            
        end
        
        TN = [TN; count_zeros/zero];
        TP = [TP; count_ones/ones];
    
    end
end