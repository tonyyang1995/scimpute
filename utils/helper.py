import numpy as np
import torch

def print_loss(loss_stat, cur_iter, total_iter, cur_epoch, total_epoch):
    message = '\n--------------------[Epoch %d/%d, Batch %d/%d]--------------------\n' % (cur_epoch, total_epoch, cur_iter, total_iter)
    for k, v in loss_stat.items():
        message += '{:>10}\t{:>10.4f}\n'.format(k, v)
    message += '--------------------------------------------------------------------\n'
    print(message)
    # return message

def print_writer(loss_stat, cur_iter, total_iter, cur_epoch, total_epoch, writer):
    counter = total_iter * (cur_epoch) + cur_iter
    for k in loss_stat:
        writer.add_scalar('train/{}'.format(k), loss_stat[k],counter)
    
def my_knn_type(img, fake_imgs, topk=10):
    # print(img.shape)
    c, h,w = img.shape
    img = img.view((h*w)).unsqueeze(0)
    out_img = img.clone().float()
    
    # print(fake_imgs.shape)
    b,c,h,w = fake_imgs.shape # 200 * 1 * 128 * 128
    fake_imgs = fake_imgs.view(b, c*h*w).detach().cpu()

    img_n = img.repeat(b, 1)
    # print(fake_imgs.shape, img_n.shape)
    diff = fake_imgs - img_n # c * (hw)
    diff = diff * diff
    rel = torch.sum(diff, dim=1) # 200 * 1
    # print(rel.argsort()[0:topk].shape)
    sim_out = fake_imgs[rel.argsort()[0:topk], :]
    # print(sim_out.shape)
    value, index = torch.median(sim_out, dim=0, keepdim=True)
    # value = torch.mean(sim_out, dim=0, keepdim=True)

    locs = (out_img == 0)
    # print(out_img.shape, value.shape)
    value = value.float()
    # return value
    # for test
    # temp = value[locs] 
    # print(len(temp[temp>0]))

    out_img[locs] = value[locs]
    # print(out_img.shape)
    return out_img
