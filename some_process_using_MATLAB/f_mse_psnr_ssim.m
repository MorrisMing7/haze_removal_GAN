function [ result ] = f_mse_psnr_ssim( x, ref )

v_ssim = ssim( x, ref );
v_psnr = psnr( x, ref );
v_mse = immse( x, ref );
result = [v_mse, v_psnr,v_ssim ];


end

