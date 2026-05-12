import torch
from torch.autograd import Variable
import os
from torchvision.utils import save_image
import numpy as np


def var(tensor, requires_grad=True):
    if torch.cuda.is_available():
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor

    var = Variable(tensor.type(dtype), requires_grad=requires_grad)

    return var


def make_img(dloader, G, z, img_num, img_size):
    if torch.cuda.is_available():
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor

    con_data, seismic_data, ground_truth = dloader.dataset[0]
    con_data = con_data.unsqueeze(dim=0)
    seismic_data = seismic_data.unsqueeze(dim=0)
    ground_truth = ground_truth.unsqueeze(dim=0)

    N = con_data.size(0)  # 1
    con_data = var(con_data.type(dtype))
    seismic_data = var(seismic_data.type(dtype))

    result_img = torch.FloatTensor(N * (img_num + 3), 1, img_size[0], img_size[1]).type(dtype)

    for i in range(N):
        result_img[i * (img_num + 3)] = con_data[i].data
        result_img[i * (img_num + 3) + 1] = seismic_data[i].data
        result_img[i * (img_num + 3) + 2] = ground_truth[i].data

        for j in range(img_num):
            con_data_ = con_data[i].unsqueeze(dim=0)
            seismic_data_ = seismic_data[i].unsqueeze(dim=0)
            z_ = z[i, j, :].unsqueeze(dim=0)

            out_img = G(con_data_, seismic_data_, z_)
            result_img[i * (img_num + 3) + j + 3] = out_img.data

    result_img = result_img / 2 + 0.5
    result_img.clamp(0, 1)

    return result_img


def save_result_img_as_txt(result_img, epoch):
    bs = result_img.shape[0]

    x = result_img.shape[2]
    y = result_img.shape[3]

    savedir = './generated_image_txt_format/epoch_%d' % (epoch + 1)

    try:
        os.makedirs('%s' % savedir)
    except OSError:
        pass

    con_data = result_img[0]
    save_image(con_data, fp='%s/con_data.png' % savedir)

    result_img_ = result_img.detach().cpu().numpy()

    for i in range(2, bs, 1):
        temp = result_img_[i].reshape((x, y))
        template = np.zeros((x, y))

        for k in range(x):
            template[x - 1 - k, :] = temp[k, :]

        template = template.reshape((x * y, 1))

        file = open('%s/fake_image_%d.txt' % (savedir, i - 1), 'w')
        file.write(str(y) + ' ' + str(x) + ' ' + str(1) + '\n')
        file.write(str(1) + '\n')
        file.write('facies' + '\n')

        for j in range(len(template)):
            if template[j] <= 0.5:
                value = int(template[j])
                file.write(str(int(value)) + '\n')
            else:
                value = 1
                file.write(str(int(value)) + '\n')

        file.close()