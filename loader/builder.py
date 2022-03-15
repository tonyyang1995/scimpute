def build_dataset(opt):
    dataset = None
    name = opt.dataset_mode
    if name == 'scigan':
        from .scigan import sciganDataset
        dataset = sciganDataset(opt)
        
    return dataset