function res = nonlin(x)
    t1 = exp(-x);
    t2 = 1 + t1;
    res = 1 ./ t2;
end