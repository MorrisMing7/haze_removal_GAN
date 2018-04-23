function [ result ] = f_trans_t(  t )
% t is between 0 and 1
    tmp =max(max(t));
    t=t/tmp*(0.5+0.4*tmp);
    result= 0.8./(1+exp(-13*t+7)).*(t<0.5385)+ (0.6*t+0.0770).*(t>=0.5385);
end

