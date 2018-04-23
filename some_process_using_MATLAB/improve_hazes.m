% from e:/data/cloud_from_cloudGAN6_001
% save_to e:/data/cloud_from_cloudGAN6_001_improve

cloud_dir = 'e:/data/cloud_from_cloudGAN6_001';
cloud_improve_dir='e:/data/cloud_from_cloudGAN6_001_improve';
clouds=dir( [cloud_dir, '/*.png' ] );
i=1;
while(i<=length(clouds))
    t=imread( [ cloud_dir, '/', clouds(i).name ] );
    t=double(t)/255;
    t2=f_trans_t(t);
    t3 = imfilter(t2, fspecial('disk',3));
    imshow( [ t,t2,t3]);
%     imwrite(t3, [cloud_improve_dir, '/', int2str(i), '.png']);
    i = i+1;
    if (mod(i,100)==0)
        fprintf('%d\n',i);
    end
end