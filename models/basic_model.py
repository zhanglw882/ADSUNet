import os

import cv2
import torch

from misc.imutils import save_image
from models.networks import *
import scipy.io as scio

class CDEvaluator():

    def __init__(self, args):

        self.n_class = args.n_class
        # define G
        self.net_G = define_G(args=args, gpu_ids=args.gpu_ids)
        self.net_G_name = args.net_G
        self.device = torch.device("cuda:%s" % args.gpu_ids[0]
                                   if torch.cuda.is_available() and len(args.gpu_ids)>0
                                   else "cpu")

        print(self.device)

        self.checkpoint_dir = args.checkpoint_dir

        self.pred_dir = args.output_folder
        os.makedirs(self.pred_dir, exist_ok=True)

    def load_checkpoint(self, checkpoint_name='best_ckpt.pt'):

        if os.path.exists(os.path.join(self.checkpoint_dir, self.net_G_name+'_'+checkpoint_name)):
            # load the entire checkpoint
            checkpoint = torch.load(os.path.join(self.checkpoint_dir, self.net_G_name+'_'+checkpoint_name),
                                    map_location=self.device)

            self.net_G.load_state_dict(checkpoint['model_G_state_dict'])
            self.net_G.to(self.device)
            # update some other states
            self.best_val_acc = checkpoint['best_val_acc']
            self.best_epoch_id = checkpoint['best_epoch_id']

        else:
            raise FileNotFoundError('no such checkpoint %s' % checkpoint_name)
        return self.net_G


    def _visualize_pred(self):
        # pred = torch.argmax(self.G_pred, dim=1, keepdim=True)

        output = self.G_pred[:,0,:,:]
        # output = torch.squeeze(self.G_pred)

        pred = torch.sigmoid(output)



        name = self.batch['name']
        for i, pred_ in enumerate(pred):
            file_name = os.path.join(
                self.pred_dir, name[i][0:-3]+'mat')

            pred_save = pred_.cpu().detach().numpy()


            # save_name = folder_name + ('%05d_%d.mat' % (i % 100 + 1, frame_num + 1))
            scio.savemat(file_name, {'Out': pred_save})



        # cv2.imshow('pred',pred.data.cpu().numpy())
        # cv2.waitKey(0)
        # pred = self.G_pred
        pred_vis = (1 - pred) * 255

        # cv2.imshow('pred_vis',pred_vis.data.cpu().numpy())
        # cv2.waitKey(0)

        return pred_vis

    def _forward_pass(self, batch):
        self.batch = batch
        img_in1 = batch['A'].to(self.device)
        img_in2 = batch['B'].to(self.device)
        self.shape_h = img_in1.shape[-2]
        self.shape_w = img_in1.shape[-1]
        self.G_pred = self.net_G(img_in1, img_in2)[-1]
        return self._visualize_pred()

    def eval(self):
        self.net_G.eval()

    def _save_predictions(self):
        """
        保存模型输出结果，二分类图像
        """
        name = self.batch['name']
        preds = self._visualize_pred()
        for i, pred in enumerate(preds):
            file_name = os.path.join(
                self.pred_dir, name[i].replace('.jpg', '.png'))
            # pred = pred[0].cpu().numpy()
            pred = pred.cpu().detach().numpy()
            save_image(pred, file_name)

