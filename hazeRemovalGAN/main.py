# coding=utf-8
from haze_removal_net import *

if __name__ == "__main__":
    train_dir ='/media/morris/文档/data/Beijing_img_X_cloudGAN6'
    test_dir = '/media/morris/文档/data/Beijing_img_X_cloudGAN6/for_test'
    for i in [train_dir,test_dir]:
        if not os.path.exists(i):
            raise Exception('train or test dir does not exist')

    log_dir = 'log'
    ckt_dir = 'ckt'
    display_dir = 'display'
    result_dir = test_dir+'/hazeRemovalGAN_result'

    for i in [log_dir,ckt_dir,display_dir,result_dir]:
        if not os.path.exists(i):
            os.makedirs(i)

    gan = haze_removal_net(im_size=512,batch_size=12,max_epoch=20,
                           gan_weight=1.0,l1_weight=90,adam_lr=2e-4,adam_beta1=0.5,
                           train_dir=train_dir,test_dir = test_dir,
                           log_dir=log_dir,ckt_dir=ckt_dir,display_dir=display_dir,
                           freq_summary=50,freq_trace=0,freq_display=500,
                           freq_process=100,freq_save=1000
                           )

    gan.train()
    print('train process is done, star to test')
    gan.have_trained=True
    gan.test(test_dir,result_dir)
