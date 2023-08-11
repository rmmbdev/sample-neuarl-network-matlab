function error = predictNN(data,label,w,w_b)
    
    err_arr = [];
    for j=1:size(data,1)
        l = cell(1,length(w));

        % forward
        l{1} = data(j,:);
        for k=1:length(w)
            l{k+1} = nonlin(l{k} * w{1,k} + w_b{k});
        end
        
        err_arr = [err_arr (label(j) - l{length(l)}).^2 ];
    end
    error = mean(err_arr);
end