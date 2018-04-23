function [ imgs  ] = f_cut_img( src_img, cut_size, stride )
% 
    imgs = {};
    [tx, ty,~] = size(src_img);
    xx=1;
    count=0;
    while(xx<=tx-cut_size+1)
        yy=1;
        while(yy<=ty-cut_size+1)
            tmp=src_img(xx:xx+cut_size-1,yy:yy+cut_size-1,:);
            count=count+1;
            imgs{count}=tmp;
            yy=yy+stride;
        end
        xx=xx+stride;
    end
  
end


