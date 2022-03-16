import argparse, os, time, sys, gc, cv2
from multiprocessing import Pool
from functools import partial
import signal
from plyfile import PlyData, PlyElement

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torch.nn.functional as F

from models import MosaicMVSNet
from utils_mosaic import *
from datasets import find_dataset_def
from datasets.data_io_mosaic import read_pfm, save_pfm, read_camera_parameters, read_img, read_mask, save_mask, read_pair_file, write_cam, write_cam2
import numpy as np

cudnn.benchmark = True

parser = argparse.ArgumentParser(description='Predict depth, filter, and fuse')
parser.add_argument('--model', default='mvsnet', help='select model')

parser.add_argument('--dataset', default='dtu_yao_eval', help='select dataset')
parser.add_argument('--testpath', help='testing data dir for some scenes')
parser.add_argument('--testpath_single_scene', help='testing data path for single scene')
parser.add_argument('--testlist', help='testing scene list')

parser.add_argument('--batch_size', type=int, default=2, help='testing batch size')
parser.add_argument('--numdepth', type=int, default=400, help='the number of depth values')

parser.add_argument('--loadckpt', default=None, help='load a specific checkpoint')
parser.add_argument('--outdir', default='./outputs', help='output dir')
parser.add_argument('--display', action='store_true', help='display depth images and masks')

parser.add_argument('--share_cr', action='store_true', help='whether share the cost volume regularization')

parser.add_argument('--ndepths', type=str, default="48,32,8", help='ndepths')
parser.add_argument('--depth_inter_r', type=str, default="4,2,1", help='depth_intervals_ratio')
parser.add_argument('--cr_base_chs', type=str, default="8,8,8", help='cost regularization base channels')
parser.add_argument('--grad_method', type=str, default="detach", choices=["detach", "undetach"], help='grad method')

parser.add_argument('--interval_scale', type=float, default=1.06,  help='the depth interval scale')
parser.add_argument('--num_view', type=int, default=5, help='num of view')
parser.add_argument('--max_h', type=int, default=864, help='testing max h')
parser.add_argument('--max_w', type=int, default=1152, help='testing max w')
parser.add_argument('--fix_res', type=bool, default=True, help='scene all using same res')

parser.add_argument('--num_worker', type=int, default=4, help='depth_filer worker')
parser.add_argument('--save_freq', type=int, default=1, help='save freq of local pcd')

# parse arguments and check
args = parser.parse_args()
print_args(args)
if args.testpath_single_scene:
    args.testpath = os.path.dirname(args.testpath_single_scene)

num_stage = len([int(nd) for nd in args.ndepths.split(",") if nd])
print("num_stage : ", num_stage)
Interval_Scale = args.interval_scale
zero_index_list=[]

def save_depth(testlist):
    for scene in testlist:
        save_scene_depth([scene])
    #print("total_zero_percentage : ",sum(zero_index_list) /len(zero_index_list))

# run CasMVS model to save depth maps and confidence maps
def save_scene_depth(testlist):
    # dataset, dataloader
    MosaicMVSDataset = find_dataset_def(args.dataset)
    test_dataset = MosaicMVSDataset(args.testpath, testlist, "test", args.num_view, args.numdepth, Interval_Scale,max_h=args.max_h, max_w=args.max_w, fix_res=args.fix_res)
    TestImgLoader = DataLoader(test_dataset, args.batch_size, shuffle=False, num_workers=4, drop_last=False)

    # model
    model = MosaicMVSNet(ndepths=[int(nd) for nd in args.ndepths.split(",") if nd],
                          depth_interals_ratio=[float(d_i) for d_i in args.depth_inter_r.split(",") if d_i],
                          share_cr=args.share_cr,
                          cr_base_chs=[int(ch) for ch in args.cr_base_chs.split(",") if ch],
                          grad_method=args.grad_method)

    # load checkpoint file specified by args.loadckpt
    print("loading model {}".format(args.loadckpt))
    state_dict = torch.load(args.loadckpt, map_location=torch.device("cpu"))
    model.load_state_dict(state_dict['model'], strict=True)
    model = nn.DataParallel(model)
    model.cuda()
    model.eval()

    with torch.no_grad():
        for batch_idx, sample in enumerate(TestImgLoader):
            sample_cuda = tocuda(sample)
            start_time = time.time()
            outputs = model(sample_cuda["imgs"], sample_cuda["proj_matrices"], sample_cuda["depth_values"])
            end_time = time.time()
            outputs = tensor2numpy(outputs)
            del sample_cuda
            filenames = sample["filename"]

            cams = sample["proj_matrices"]["stage{}".format(num_stage)].numpy()
            imgs = sample["imgs"].numpy()

            print('Iter {}/{}, Time:{} Res:{}'.format(batch_idx, len(TestImgLoader), end_time - start_time,
                                                      imgs[0].shape))
            #zero_index_list.append(outputs["zero_index"])

            # save depth maps and confidence maps
            for filename, cam, img, depth_est, photometric_confidence in zip(filenames, cams, imgs, \
                                                                             outputs["depth"],
                                                                             outputs["photometric_confidence"]):
                img = img[0]  # ref view
                cam = cam[0]  # ref cam
                depth_filename = os.path.join(args.outdir, filename.format('depth_est', '.pfm'))
                confidence_filename = os.path.join(args.outdir, filename.format('confidence', '.pfm'))
                cam_filename = os.path.join(args.outdir, filename.format('cams', '_cam.txt'))
                img_filename = os.path.join(args.outdir, filename.format('images', '.jpg'))
                ply_filename = os.path.join(args.outdir, filename.format('ply_local', '.ply'))
                os.makedirs(depth_filename.rsplit('/', 1)[0], exist_ok=True)
                os.makedirs(confidence_filename.rsplit('/', 1)[0], exist_ok=True)
                os.makedirs(cam_filename.rsplit('/', 1)[0], exist_ok=True)
                os.makedirs(img_filename.rsplit('/', 1)[0], exist_ok=True)
                os.makedirs(ply_filename.rsplit('/', 1)[0], exist_ok=True)
                # save depth maps
                save_pfm(depth_filename, depth_est)
                # save confidence maps
                save_pfm(confidence_filename, photometric_confidence)
                # save cams, img
                write_cam(cam_filename, cam)
                img = np.clip(np.transpose(img, (1, 2, 0)) * 255, 0, 255).astype(np.uint8)
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(img_filename, img_bgr)

                if num_stage == 1:
                    downsample_img = cv2.resize(img, (int(img.shape[1] * 0.25), int(img.shape[0] * 0.25)))
                elif num_stage == 2:
                    downsample_img = cv2.resize(img, (int(img.shape[1] * 0.5), int(img.shape[0] * 0.5)))
                elif num_stage == 3:
                    downsample_img = img

                if batch_idx % args.save_freq == 0:
                    generate_pointcloud(downsample_img, depth_est, ply_filename, cam[1, :3, :3])
        #print("total_zero_mean :",sum(zero_index_list)/len(zero_index_list))
        #path=""
        #write_cam2(path,zero_index_list)
    torch.cuda.empty_cache()
    gc.collect()


def init_worker():
    '''
    Catch Ctrl+C signal to termiante workers
    '''
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def pcd_filter_worker(scan):
    if args.testlist != "all":
        print(scan)
        scan_id = int(scan[5:])
        save_name = 'mvsnet{:0>3}_l3.ply'.format(scan_id)
    else:
        save_name = '{}.ply'.format(scan)
    pair_folder = os.path.join(args.testpath, scan)
    scan_folder = os.path.join(args.outdir, scan)
    out_folder = os.path.join(args.outdir, scan)
    filter_depth(pair_folder, scan_folder, out_folder, os.path.join(args.outdir, save_name))


def pcd_filter(testlist, number_worker):
    partial_func = partial(pcd_filter_worker)

    p = Pool(number_worker, init_worker)
    try:
        p.map(partial_func, testlist)
    except KeyboardInterrupt:
        print("....\nCaught KeyboardInterrupt, terminating workers")
        p.terminate()
    else:
        p.close()
    p.join()


if __name__ == '__main__':
    if args.testlist != "all":
        with open(args.testlist) as f:
            content = f.readlines()
            testlist = [line.rstrip() for line in content]

    save_depth(testlist)
