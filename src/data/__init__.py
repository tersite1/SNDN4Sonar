from .det import load_det_data

def load_data(opt):
    task = opt.get('task', 'det').lower()
    print('Data path: {}'.format(opt['data']['path']))
    if task != 'det':
        raise ValueError(f"Only 'det' task is supported, got '{task}'")
    return load_det_data(opt)
