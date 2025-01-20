from argparse import ArgumentParser
import torch
from models.trainer import *

print(torch.cuda.is_available())

"""
the main function for training the CD networks
训练代码
"""

# 训练代码
def train(args):
    dataloaders = utils.get_loaders(args) ##导入数据集
    model = CDTrainer(args=args, dataloaders=dataloaders)
    model.train_models()


# 测试代码
def test(args):
    from models.evaluator import CDEvaluator
    dataloader = utils.get_loader(args.data_name, img_size=args.img_size,
                                  batch_size=args.batch_size, is_train=False,
                                  split='test')
    model = CDEvaluator(args=args, dataloader=dataloader)

    model.eval_models()


if __name__ == '__main__':
    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--project_name', default='ADSUNet', type=str)
    parser.add_argument('--checkpoint_root', default='checkpoints', type=str)
    parser.add_argument('--vis_root', default='vis', type=str)

    # data
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--dataset', default='IRSTD', type=str)
    parser.add_argument('--data_name', default='inter_frame_data', type=str)

    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--split', default="train", type=str) ####train-val-demo(test)
    parser.add_argument('--split_val', default="val", type=str)

    parser.add_argument('--img_size', default=256, type=int)
    parser.add_argument('--shuffle_AB', default=False, type=str)

    # model
    parser.add_argument('--n_class', default=2, type=int)
    parser.add_argument('--embed_dim', default=64, type=int)
    parser.add_argument('--pretrain', default=None, type=str) ##预训练权重
    parser.add_argument('--multi_scale_train', default=False, type=str)
    parser.add_argument('--multi_scale_infer', default=False, type=str)
    parser.add_argument('--multi_pred_weights', nargs = '+', type = float, default = [0.5, 0.5, 0.5, 0.8, 1.0])

    # 网络模型
    parser.add_argument('--net_G', default='SiamUnet_conc_diff_cbam', type=str,
                        help='')
    parser.add_argument('--loss', default='miou', type=str)

    # optimizer
    parser.add_argument('--optimizer', default='sgd', type=str)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--max_epochs', default=1000, type=int)
    parser.add_argument('--lr_policy', default='linear', type=str,
                        help='linear | step')
    parser.add_argument('--lr_decay_iters', default=100, type=int)


    args = parser.parse_args()
    utils.get_device(args) ##GPU数量
    print(args.gpu_ids)
    
    #  checkpoints dir
    args.checkpoint_dir = os.path.join(args.checkpoint_root, args.project_name) ##权重保存的路径
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    #  visualize dir
    args.vis_dir = os.path.join(args.vis_root, args.project_name) ##可视化保存的路径
    os.makedirs(args.vis_dir, exist_ok=True)

    train(args)

    test(args)
