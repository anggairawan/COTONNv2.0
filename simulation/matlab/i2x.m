function x = i2x(id, ipd, eta, ll)
    tmp = size(ipd);
    dim = tmp(2);

    % id to space
    i = dim;
    x = zeros(1, dim);
    while (i>1)
        num = floor(id/ipd(i));
        id = mod(id, ipd(i));
        x(i) = ll(i)+num*eta(i);
        
        i = i - 1;
    end
    num = id;
    x(1) = ll(i)+num*eta(1);
end