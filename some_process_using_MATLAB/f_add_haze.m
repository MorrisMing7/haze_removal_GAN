function [ I ] = f_add_haze(J , t, A)
% j is a picture of uint8
%t is a double matrix of value between0,1 represent the thickness of cloud
t=1-t;      %transform to opacity
I=zeros(size(J));
J=double(J);
I(:,:,1)=J(:,:,1).*t + A*(1-t);
I(:,:,2)=J(:,:,2).*t + A*(1-t);
I(:,:,3)=J(:,:,3).*t + A*(1-t);
I=uint8(I);

end