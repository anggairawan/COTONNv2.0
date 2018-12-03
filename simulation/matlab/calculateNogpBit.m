function ngpbit = calculateNogpBit(eta, ll, ur)
    tmp = size(eta);
    dim = tmp(2);

    ngpbit = zeros(1, dim);

    for i = 1:dim
        ngpbit(i) = (ur(i)-ll(i))/eta(i) + 1;
        ngp_bin = dec2bin(ngpbit(i));
        size_bin = size(ngp_bin);
        ngpbit(i) = size_bin(2);
    end
end

