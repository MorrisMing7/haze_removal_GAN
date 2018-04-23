test_dir = 'E:\data\Beijing_img_X_cloudGAN6\for_test';
dcp_dir = [ test_dir, '\dcp_result' ];
cnn_dir = [ test_dir, '\cnn_result' ];
gan_dir = [ test_dir, '\hazeRemovalGAN_result____old' ];
gan_list=dir( [ gan_dir, '/*.png' ] );
l_gan_list = length(gan_list);
v_dcp = zeros( l_gan_list, 3);
v_cnn = zeros( l_gan_list, 3);
v_gan = zeros( l_gan_list, 3);

for i=1: l_gan_list
    name = gan_list(i).name;
    result_dcp = imread( [ dcp_dir , '/' name ] );
    result_dcp = result_dcp( :, end-511:end, : );
    result_cnn = imread( [ cnn_dir , '/' name ] );
    result_cnn = result_cnn( :, end-511:end, : );
    result_gan = imread( [ gan_dir , '/' name ] );
    result_gan = result_gan( :, 518:1029, : );
    test = imread( [ test_dir , '/', name ] );
    target = test( :, 1:512, : );
    input = test(:, 513:end, :);
    imshow( [ input, result_dcp, result_cnn, result_gan, target ]);
%     
%     v_dcp(i, : ) = f_mse_psnr_ssim(result_dcp, target);
%     v_cnn(i, : ) = f_mse_psnr_ssim(result_cnn, target);
%     v_gan(i, : ) =  f_mse_psnr_ssim(result_gan, target);
    fprintf('%d \n',i);
end
result = zeros(3,3);
result(1,:)=mean(v_dcp);
result(2,:)=mean(v_cnn);
result(3,:)=mean(v_gan);

