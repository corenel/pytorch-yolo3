from darknet import Darknet
import dataset
import torch
from torch.autograd import Variable
from torchvision import datasets, transforms
from utils import *
import os

def valid(datacfg, cfgfile, weightfile, outfile):
    options = read_data_cfg(datacfg)
    valid_images = options['valid']
    name_list = options['names']
    prefix = 'results'
    names = load_class_names(name_list)

    with open(valid_images) as fp:
        tmp_files = fp.readlines()
        valid_files = [item.rstrip() for item in tmp_files]

    m = Darknet(cfgfile)
    m.print_network()
    m.load_weights(weightfile)
    m.cuda()
    m.eval()
    # FIXME modify number of classes
    num_classes = 1
    min_box_scale = 8. / m.width

    valid_dataset = dataset.listDataset(valid_images, shape=(m.width, m.height),
                       shuffle=False,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                       ]))
    valid_batchsize = 2
    assert(valid_batchsize > 1)

    kwargs = {'num_workers': 4, 'pin_memory': True}
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=valid_batchsize, shuffle=False, **kwargs)

    fps = [0]*num_classes
    gts = [0]*num_classes
    if not os.path.exists('results'):
        os.mkdir('results')
    if not os.path.exists('results/predicted'):
        os.mkdir('results/predicted')
    if not os.path.exists('results/ground-truth'):
        os.mkdir('results/ground-truth')
    for i in range(num_classes):
        buf = '%s/%s%s.txt' % (prefix, outfile, names[i])
        fps[i] = open(buf, 'w')
        gtbuf = '%s/%s%s_gt.txt' % (prefix, outfile, names[i])
        gts[i] = open(gtbuf, 'w')

    lineId = -1

    conf_thresh = 0.005
    nms_thresh = 0.45
    for batch_idx, (data, target) in enumerate(valid_loader):
        data = data.cuda()
        data = Variable(data, volatile = True)
        output = m(data)
        batch_boxes = output
        for i in range(int(data.shape[0])):
            lineId = lineId + 1
            fileId = os.path.basename(valid_files[lineId]).split('.')[0]
            # FIXME modify image size
            # width, height = get_image_size(valid_files[lineId])
            width, height = 960, 576
            print(valid_files[lineId])
            boxes = batch_boxes[0][i] + batch_boxes[1][i] + batch_boxes[2][i]
            boxes = nms(boxes, nms_thresh)
            for box in boxes:
                x1 = (box[0] - box[2]/2.0) * width
                y1 = (box[1] - box[3]/2.0) * height
                x2 = (box[0] + box[2]/2.0) * width
                y2 = (box[1] + box[3]/2.0) * height

                det_conf = box[4]
                with open('results/predicted/%s.txt' % fileId, 'w') as f:
                    for j in range((len(box)-5)/2):
                        cls_conf = box[5+2*j]
                        cls_id = box[6+2*j]
                        prob =det_conf * cls_conf
                        f.write('%s %f %d %d %d %d\n' % (names[cls_id], prob, x1, y1, x2, y2))

            img_path = valid_files[lineId]
            lab_path = img_path.replace('images', 'labels')
            lab_path = lab_path.replace('JPEGImages', 'labels')
            lab_path = lab_path.replace('.jpg', '.txt').replace('.png', '.txt')
            truths = read_truths_args(lab_path, min_box_scale)

            with open('results/ground-truth/%s.txt' % fileId, 'w') as f:
                for truth in truths:
                    x1 = (truth[1] - truth[3]/2.0) * width
                    y1 = (truth[2] - truth[4]/2.0) * height
                    x2 = (truth[1] + truth[3]/2.0) * width
                    y2 = (truth[2] + truth[4]/2.0) * height
                    f.write('%s %d %d %d %d\n' % (names[int(truth[0])], x1, y1, x2, y2))

    for i in range(num_classes):
        fps[i].close()

if __name__ == '__main__':
    import sys
    if len(sys.argv) == 4:
        datacfg = sys.argv[1]
        cfgfile = sys.argv[2]
        weightfile = sys.argv[3]
        outfile = 'comp4_det_test_'
        valid(datacfg, cfgfile, weightfile, outfile)
    else:
        print('Usage:')
        print(' python valid.py datacfg cfgfile weightfile')
