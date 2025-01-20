
# 针对自建的多帧数据集进行网络的推理
from argparse import ArgumentParser
import utils
import torch
from models.basic_model import CDEvaluator
import os
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF
import scipy.io as scio
import cv2
import time
"""
quick start
sample files in ./samples
save prediction files in the ./samples/predict
"""

def get_args():
    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument('--project_name', default='ADSUNet', type=str)
    parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--checkpoint_root', default=r'./checkpoints', type=str)
    parser.add_argument('--output_folder', default='ISTD_data/result', type=str)

    # data
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--dataset', default='CDDataset', type=str)
    parser.add_argument('--data_name', default='DSIFN', type=str)

    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--split', default="demo", type=str)
    parser.add_argument('--img_size', default=256, type=int)

    # model
    parser.add_argument('--n_class', default=2, type=int)
    parser.add_argument('--embed_dim', default=64, type=int)
    parser.add_argument('--net_G', default='SiamUnet_conc_diff_cbam', type=str,
                        help='')
    parser.add_argument('--checkpoint_name', default='best_ckpt.pt', type=str)

    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = get_args()
    utils.get_device(args)
    device = torch.device("cuda:%s" % args.gpu_ids[0]
                          if torch.cuda.is_available() and len(args.gpu_ids)>0
                        else "cpu")
    args.checkpoint_dir = os.path.join(args.checkpoint_root, args.project_name)
    os.makedirs(args.output_folder, exist_ok=True)

    log_path = os.path.join(args.output_folder, 'log_vis.txt')

    model = CDEvaluator(args)
    model.load_checkpoint(args.checkpoint_name)
    time_all = 0
    with torch.no_grad():
        model.eval()

        seqs_names = ['Sequence1']
        path = r'./testdata'
        for o,seqs_name in enumerate(seqs_names):


            A_path = os.path.join(path,seqs_name+'/new/A/')
            B_path = os.path.join(path,seqs_name+'/new/B/')

            file_names = os.listdir(A_path)

            for _,file_name in enumerate(file_names):
                img_A_path = os.path.join(A_path,file_name)
                img_B_path = os.path.join(B_path,file_name)

                img_A = np.asarray(Image.open(img_A_path).convert('RGB'))
                img_B = np.asarray(Image.open(img_B_path).convert('RGB'))

                img_in1 = TF.to_tensor(img_A)
                img_in2 = TF.to_tensor(img_B)

                # img_in1 = TF.normalize(img_in1, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                # img_in2 = TF.normalize(img_in2, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

                img_in1 = img_in1.unsqueeze(dim=0)
                img_in2 = img_in2.unsqueeze(dim=0)

                img_in1 = img_in1.to(device)
                img_in2 = img_in2.to(device)

                ori_size = img_in1.size()

                time_1 = time.time()
                model.G_pred = model.net_G(img_in1, img_in2)[-1]
                time_2 = time.time()
                time_3 = time_2 - time_1
                time_all += time_3

                if len(model.G_pred.size()) == 4:
                    output = model.G_pred[:, 0, :, :]####
                else:
                    output = model.G_pred[0, :, :]

                pred = output.squeeze()
                output_max = torch.max(output.squeeze()[5:-5,5:-5])

                pred[0:4, :] = output_max
                pred[-4:, :] = output_max
                pred[:, 0:4] = output_max
                pred[:, -4:] = output_max

                pred = torch.sigmoid(pred)

                file_name = os.path.join(model.pred_dir+'/'+seqs_name, file_name[0:-3]+'mat')

                save_path = model.pred_dir+'/'+seqs_name
                if not os.path.exists(save_path):
                    os.mkdir(save_path)

                # file_name
                pred_save = pred.squeeze().cpu().detach().numpy()

                ## 显示图片
                cv2.imshow('pred_save', 1-pred_save)
                cv2.waitKey(0)
                # save_name = folder_name + ('%05d_%d.mat' % (i % 100 + 1, frame_num + 1))

                scio.savemat(file_name, {'Out': pred_save})
                torch.cuda.empty_cache()


        print(time_all)
