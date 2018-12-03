function i = x2i(x, ipd, eta, ll)
    tmp = size(ipd);
    dim = tmp(2);

    % space to id
    i = 0;
    for j = 1:dim
       d_id = x(j) - ll(j);
       i = i + floor((d_id+eta(j)/2.0)/eta(j))*ipd(j);
    end
    