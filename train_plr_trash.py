import argparse
import os
import torch
from dataset import PLR, get_dataloader
from model import PointPillars

def main(args):
    train_dataset = PLR(data_root=args.data_root, split='feature')
    train_dataloader = get_dataloader(dataset=train_dataset, 
                                      batch_size=args.batch_size, 
                                      num_workers=args.num_workers,
                                      shuffle=True)

    if not args.no_cuda:
        pointpillars = PointPillars(nclasses=args.nclasses).cuda()
    else:
        pointpillars = PointPillars(nclasses=args.nclasses)

    for i, data_dict in enumerate(train_dataloader):
        if not args.no_cuda:
            # move the tensors to the cuda
            for key in data_dict:
                for j, item in enumerate(data_dict[key]):
                    if torch.is_tensor(item):
                        data_dict[key][j] = data_dict[key][j].cuda()

        batched_pts = data_dict['batched_pts']
        batched_gt_bboxes = data_dict['batched_gt_bboxes']
        batched_labels = data_dict['batched_labels']
        feature_x = \
            pointpillars(batched_pts=batched_pts, 
                         mode='feature',
                         batched_gt_bboxes=batched_gt_bboxes, 
                         batched_gt_labels=batched_labels)

        # Rest of the code here...

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Configuration Parameters')
    parser.add_argument('--data_root', default='/home/hojkim/dataset', 
                        help='your data root for kitti')
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--nclasses', type=int, default=3)
    parser.add_argument('--no_cuda', action='store_true',
                        help='whether to use cuda')
    args = parser.parse_args()

    main(args)
