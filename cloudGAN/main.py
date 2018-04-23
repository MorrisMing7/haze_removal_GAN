from sim import *


if __name__=="__main__":
    img_dir = '../00cloud'
    display_dir = './display'
    samples_dir = './samples'
    log_dir = './log'
    ckpt_dir = './ckpt'
    for i in [display_dir,samples_dir,log_dir,ckpt_dir]:
        if not os.path.exists(i):
            os.mkdir(i)

    wgan = CloudGAN(real_img_dir=img_dir ,display_dir=display_dir,
                    sample_dir=samples_dir,log_dir=log_dir,ckpt_dir=ckpt_dir,

                    )

    # wgan.train()
    wgan.test(1000)