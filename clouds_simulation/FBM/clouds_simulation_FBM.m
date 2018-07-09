function  [ r ] = clouds_simulation_FBM( x,  rand_method, a, b )
    x=mapminmax(x,-1,1);
    width = size(x,1);
    if width==2
        r=ones(3,3);
        r(2,2)=sum(x(:))/4;
        r(1,1)=x(1,1);
        r(1,3)=x(1,2);
        r(3,1)=x(2,1);
        r(3,3)=x(2,2);
        r(1,2)=( x(1,1)+x(1,2)+r(2,2) )/4;
        r(2,1)=( x(1,1)+x(2,1)+r(2,2) )/4;
        r(2,3)=( x(1,2)+x(2,2)+r(2,2) )/4;
        r(3,2)=( x(2,1)+x(2,2)+r(2,2) )/4;
    elseif mod(width,2)==0
        r=ones(width+1,width+1);
        for i=1:width+1
            r(width+1,i)=r(width,i)                                         +random(rand_method,a,b) ;
            r(i,width+1)=r(i,width)                                         +random(rand_method,a,b) ;
        end
    else
        w=2*width-1;
        r = zeros(w,w);
        for i=1:width
            for j=1:width
                r(2*(i-1)+1,2*(j-1)+1)=x(i,j);
            end
        end
        for i=1:width-1
            for j=1:width-1
                r(2*i,2*j)=( x(i,j)+x(i+1,j)+x(i,j+1)+x(i+1,j+1) )/4    +random(rand_method,a,b) ;
            end
        end
        for i=1:width-1
            r(2*i,1)=( r(2*i,2)+r(2*i-1,1)+r(2*i+1,1) )/3               +random(rand_method,a,b);
            r(2*i,w)=( r(2*i,w-1)+r(2*i-1,w)+r(2*i+1,w) )/3         +random(rand_method,a,b);
            r(1,2*i)=( r(2,2*i)+r(1,2*i-1)+r(1,2*i+1) )/3               +random(rand_method,a,b);
            r(w,2*i)=( r(w-1,2*i)+r(w,2*i-1)+r(w,2*i+1) )/3         +random(rand_method,a,b);
        end
        for i = 2:2:w-1
            for j = 3:2:w-1
                r(i,j)=(r(i-1,j)+r(i,j-1)+r(i+1,j)+r(i,j+1))/4             +random(rand_method,a,b);
            end
        end
        for j = 2:2:w-1
            for i = 3:2:w-1
                r(i,j)=(r(i-1,j)+r(i,j-1)+r(i+1,j)+r(i,j+1))/4              +random(rand_method,a,b);
            end
        end
    end
    minr=min(r(:));
    r=(r-minr)./(max(r(:))-minr);
end