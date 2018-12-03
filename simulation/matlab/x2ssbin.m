function bin = x2ssbin(x, ipd, eta, ll, ur, inputs)
    tmp = size(ipd);
    dim = tmp(2);

    % space to id
    id = 0;
    for i = 1:dim
       d_id = x(i) - ll(i);
       id = id + floor((d_id+eta(i)/2.0)/eta(i))*ipd(i);
    end
    
    % to ss
    ss = abstractionToCoordinate(id, ipd);
    
    raw_bin = char.empty;
    ngpbit = calculateNogpBit(eta, ll, ur);
    % id to binary
    for i = 1:dim
        raw_bin = [raw_bin dec2bin(ss(i), ngpbit(i))];
    end
    
    bin = zeros(1, inputs);

    for i=1:inputs
        bin(i) = str2num(raw_bin(i));
    end 
end