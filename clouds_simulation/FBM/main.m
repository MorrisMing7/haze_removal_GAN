r=[0,1;0.5,0.8];
rand_method='norm';
a=0;
b=0.3;
for i=1:10
    r=clouds_simulation_FBM(r,rand_method,a,b);
    imshow(r);
end





