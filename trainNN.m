function [w,w_b] = trainNN(data,label,epoch,eta,specs)

    layers_count = length(specs);
    
    w = cell(1,layers_count - 1);    
    for i = 1:layers_count - 1
       w{1,i} = unifrnd(-1,1, specs(i),specs(i+1));
    end
    
    w_b = cell(1,layers_count - 1);
    unifrnd(-1,1,1,layers_count - 1);
    for i = 1:layers_count - 1
       w_b{1,i} = unifrnd(-1,1, 1,specs(i+1));
    end
    
    
    for i=1:epoch
        for j=1:size(data,1)
            l = cell(1,layers_count);
            
            % forward
            l{1} = data(j,:);
            for k=1:length(w)
                l{k+1} = nonlin(l{k} * w{1,k} + w_b{k});
            end  
            
            % delta
            d = cell(1,length(w));
            d{length(d)} = (label(j,1) - l{length(l)}) .* nonlinDer(l{length(l)});
            for k=length(d) - 1:-1:1
                d{k} = (w{k+1} * d{k+1}')' .* nonlinDer(l{k + 1});
            end
            
            % update
            for k=1:length(w)                
                w{k} = w{k} + eta .* (l{k}' * d{k});
                w_b{k} = w_b{k} + eta .* (nonlin(1 * w_b{k}) .* d{k});
            end
        end        
    end
end