import os
import time
import torch
import numpy as np

from options.sciganOption import sciGANopt
from loader.builder import build_dataset
from model.sciGANmodel import sciGANModel
from tensorboardX import SummaryWriter
from utils.helper import print_loss, print_writer, my_knn_type 


def infer(opt):
    dataset = build_dataset(opt)
    # dataloader = torch.utils.data.DataLoader(
    #     dataset,
    #     batch_size=1,
    #     shuffle=False
    # )

    ganmodel = sciGANModel(opt)
    ganmodel.load(opt.load_g, opt.load_d)

    # for each spot, generate 200 fake imgs
    fake_imgs = []

    z = np.random.normal(0, 1, (200, opt.latent_dim))
    z = torch.from_numpy(z)
    ganmodel.set_input({'z': z}, mode='infer')
    fake_imgs = ganmodel.inference()

    data = open('dataset/PDAC/GSM3036911_PDAC-A-ST1-filtered.txt').readlines()

    impute = open('results/PADC/impute.csv', 'w')
    header = data[0]
    impute.write(header)

    for k in range(len(dataset)):
        img, _ = dataset[k]
        rels = my_knn_type(img, fake_imgs, topk=10)
        conts = data[k+1].strip().split('\t')
        res = [conts[0]]
        for num in range(rels.shape[1]): # 1 * hw
            res.append(str(int(rels[0, num])))
        msg = '\t'.join(res)
        impute.write(msg+'\n')

    impute.close()



def train(opt):
    expr_dir = os.path.join(opt.checkpoint_dir, opt.name)
    writer = SummaryWriter(os.path.join(expr_dir,'train'))
    dataset = build_dataset(opt)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=True
    )

    ganmodel = sciGANModel(opt)
    for epoch in range(opt.epochs):
        for bi, (img, z) in enumerate(dataloader):
            ganmodel.set_input({'img': img.float(), 'z': z.float()}, mode='train')
            ganmodel.update_optimzer()
            if bi % 5 == 0:
                loss = ganmodel.get_cur_loss()
                print_loss(loss, bi, len(dataloader), epoch, opt.epochs)
                print_writer(loss, bi, len(dataloader), epoch, opt.epochs, writer)
        if epoch % 20 == 0:
            ganmodel.save(epoch)

if __name__ == '__main__':
    opt = sciGANopt().parse()
    # train(opt)
    infer(opt)
