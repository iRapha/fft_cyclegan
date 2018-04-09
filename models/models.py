def create_model(opt):
    model = None
    print(opt.model)
    if opt.model == 'cycle_gan':
        assert(opt.dataset_mode == 'unaligned')
        from .cycle_gan_model import CycleGANModel
        model = CycleGANModel()
    elif opt.model == 'pix2pix':
        assert(opt.dataset_mode == 'aligned')
        from .pix2pix_model import Pix2PixModel
        model = Pix2PixModel()
    elif opt.model == 'test':
        assert(opt.dataset_mode == 'single')
        from .test_model import TestModel
        model = TestModel()
    elif opt.model == 'recycle_gan':
        assert(opt.dataset_mode == 'unaligned')
        from .recycle_gan_model import ReCycleGANModel
        model = ReCycleGANModel()
    elif opt.model == 'tv_gan':
        assert(opt.dataset_mode == 'unaligned')
        from .tv_gan_model import TVCycleGANModel
        model = TVCycleGANModel()
    else:
        raise ValueError("Model [%s] not recognized." % opt.model)
    model.initialize(opt)
    print("model [%s] was created" % (model.name()))
    return model
