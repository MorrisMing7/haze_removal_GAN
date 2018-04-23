save_dir = 'e:/data/Beijing_img_X_cloudGAN6';

clear_dir = 'e:/data/cut_by_hand/result';
clear_imgs = dir( [ clear_dir, '/*.jpg']);

haze_dir = 'e:/data/cloud_from_cloudGAN6_001_improve';
hazes = dir( [ haze_dir, '/*.png' ] );

i = 1;
haze_idx = 1;
while( i<=length(clear_imgs))
    tj = imread( [ clear_dir, '/', clear_imgs(i).name ] );
    j = 1;
    while(j<=5)
        tt = imread( [ haze_dir, '/', hazes(haze_idx).name ] );
        tt = imresize(tt, [512+16,512+16]);
        tt=tt(8:519,8:519);
        tt = double(tt)./255;
        ti = f_add_haze(tj, tt, 250);
%         ttt=cat(3,tt,tt,tt);
%         imshow( [ tj, ti ,uint8(ttt.*255) ] );
        imwrite( [ tj, ti ] , [ save_dir, '/', hazes(haze_idx).name ] );
        haze_idx = haze_idx+1;
        j = j+1;
    end
    if(random('Normal',0,1)<0.1)
        imwrite( [tj, tj] ,  [ save_dir, '/', hazes(haze_idx).name ] );
    end
    i =i+1;
    if(mod(i,40)==0)
        fprintf('%d imgs done\n',haze_idx);
    end
end
