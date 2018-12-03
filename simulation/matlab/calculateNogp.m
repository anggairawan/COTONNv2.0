function ngp = calculateNogp(eta, ll, ur)
    tmp = size(eta);
    dim = tmp(2);

    ngp = zeros(1, dim);

    for i = 1:dim
        ngp(i) = (ur(i)-ll(i))/eta(i) + 1;
        ngp_bin = dec2bin(ngp(i));
        size_bin = size(ngp_bin);
        ngp(i) = size_bin(2);
    end
end

