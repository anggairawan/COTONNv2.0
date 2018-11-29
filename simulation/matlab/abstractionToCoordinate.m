function ss = abstractionToCoordinate(s, no_gp)
    tmp = size(no_gp);
    dim = tmp(2);
    
    ss = ones(1, dim);

    for i = dim:-1:1
        num = fix(s/no_gp(i));
        s = mod(s,no_gp(i));
        ss(i) = num;   
    end
end