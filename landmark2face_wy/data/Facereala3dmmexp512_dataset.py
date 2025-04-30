import os.path
import random
from data.base_dataset import BaseDataset, get_params, get_transform
import torchvision.transforms as transforms
from data.image_folder import make_dataset
from PIL import Image, ImageEnhance
import numpy as np
import cv2
import torch
import time

def get_idts(config_name):
    idts = list()
    with open(os.path.join('../config', config_name + '.txt')) as f:
        for line in f:
            line = line.strip()
            video_name = line.split(':')[0]
            idts.append(video_name)
    return idts


def obtain_seq_index(index, num_frames):
    seq = list(range(index - 13, index + 13 + 1))
    seq = [min(max(item, 0), num_frames - 1) for item in seq]
    return seq

def get_3dmm_feature(img_path, idx, new_dict):
    id = img_path.split('/')[-3]
    features = new_dict[id]
    idx_list = obtain_seq_index(idx, features.shape[0])
    feature = features[idx_list, 80:144]
#    feature[:, -1] = 50
    return np.transpose(feature, (1, 0))



class Facereala3dmmexp512Dataset(BaseDataset):
    def __init__(self, opt, mode=None):
        BaseDataset.__init__(self, opt)
        img_size = opt.img_size
        idts = get_idts(opt.name.split('_')[0])
        print("---------load data list--------: ", idts)
        self.new_dict = {}
        if mode == 'train':
            self.labels = []
            self.label_starts = []
            self.label_ends = []
            count = 0
            for idt_name in idts:
                # root = '../AnnVI/feature/{}'.format(idt_name)
                root = os.path.join(opt.feature_path, idt_name)
                feature = np.load(os.path.join(root, '%s.npy' % opt.audio_feature))
                self.new_dict[idt_name] = feature
                if opt.audio_feature == "3dmm":
                    training_data_path = os.path.join(root, '{}_{}.t7'.format(img_size, mode))
                else:
                    training_data_path = os.path.join(root, '{}_{}_{}.t7'.format(img_size, mode, opt.audio_feature))
                training_data = torch.load(training_data_path)
                img_paths = training_data['img_paths']
                features_3dmm = training_data['features_3dmm']
                index = [i[0].split('/')[-1] for i in img_paths]

                image_dir = '{}/{}_dlib_crop'.format(root, img_size)
                self.label_starts.append(count)
                for img in range(len(index)):
                    img_path = os.path.join(image_dir, index[img])
                    # idx_list = obtain_seq_index(img, feature.shape[0])
                    # self.labels.append([img_path, np.transpose(feature[idx_list, ...], (1, 0))])
                    self.labels.append([img_path, features_3dmm[img]])
                    count = count + 1
                self.label_ends.append(count)

            self.label_starts = np.array(self.label_starts)
            self.label_ends = np.array(self.label_ends)
            self.transforms_image = transforms.Compose([transforms.ToTensor(),
                                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

            self.transforms_label = transforms.Compose([transforms.ToTensor(),
                                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            self.shuffle()
        elif mode == 'test':
            self.labels = []
            self.label_starts = []
            self.label_ends = []
            count = 0
            for idt_name in idts:
                # root = '../AnnVI/feature/{}'.format(idt_name)
                root = os.path.join(opt.feature_path, idt_name)
                feature = np.load(os.path.join(root, '%s.npy' % opt.audio_feature))
                self.new_dict[idt_name] = feature
                if opt.audio_feature == "3dmm":
                    training_data_path = os.path.join(root, '{}_{}.t7'.format(img_size, mode))
                else:
                    training_data_path = os.path.join(root, '{}_{}_{}.t7'.format(img_size, mode, opt.audio_feature))
                training_data = torch.load(training_data_path)
                img_paths = training_data['img_paths']
                features_3dmm = training_data['features_3dmm']
                index = [i[0].split('/')[-1] for i in img_paths]

                image_dir = '{}/{}_dlib_crop'.format(root, img_size)
                self.label_starts.append(count)
                for img in range(len(index)):
                    img_path = os.path.join(image_dir, index[img])
                    # idx_list = obtain_seq_index(img, feature.shape[0])
                    # self.labels.append([img_path, np.transpose(feature[idx_list, ...], (1, 0))])
                    self.labels.append([img_path, features_3dmm[img]])
                    count = count + 1
                self.label_ends.append(count)

            self.label_starts = np.array(self.label_starts)
            self.label_ends = np.array(self.label_ends)
            self.transforms_image = transforms.Compose([transforms.ToTensor(),
                                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

            self.transforms_label = transforms.Compose([transforms.ToTensor(),
                                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            self.shuffle()

    def shuffle(self):
        self.labels_index = list(range(len(self.labels)))
        random.shuffle(self.labels_index)

    def add_mouth_mask2(self, img):
        mask = np.ones_like(img)
        rect_area = [img.shape[1] // 2 - 60, np.random.randint(226, 246), 30, 256 - 30]
        mask_rect_area = mask[rect_area[0]: rect_area[1], rect_area[2]:rect_area[3]]
        x = np.tile(np.arange(rect_area[1] - rect_area[0])[:, np.newaxis], (1, rect_area[3] - rect_area[2]))
        x = np.flip(x, 0)
        y = np.tile(np.arange(rect_area[3] - rect_area[2])[:, np.newaxis], (1, rect_area[1] - rect_area[0])).transpose()
        zz1 = -y - x + 88 > 0
        zz2 = np.flip(zz1, 1)
        zz = (zz1 + zz2) > 0
        mask[rect_area[0]:rect_area[1], rect_area[2]:rect_area[3]] = np.tile(zz[:, :, np.newaxis], (1, 1, 3)) * 1
        imgm = img * mask
        return imgm

    def __getitem__(self, index):
        # s1= time.time()
        idx = self.labels_index[index]
        img_path, feature_3dmm_idx= self.labels[idx]
       # print(img_path, feature_3dmm_idx)
        feature_3dmm = get_3dmm_feature(img_path, feature_3dmm_idx, self.new_dict)
        #print(img_path, feature_3dmm_idx, feature_3dmm.shape)

        img = np.array(Image.open(img_path).convert('RGB'))
        img = np.array(np.clip(img + np.random.randint(-20, 20, size=3, dtype='int8'), 0, 255), dtype='uint8')
        cut_pad1 = np.random.randint(0, 20)
        cut_pad2 = np.random.randint(0, 20)
        img = img[cut_pad1:512 + cut_pad1, cut_pad2:512 + cut_pad2]
        # s2 =time.time()
        # print('get data and read data ', s2-s1)
        mask_B = img.copy()
        # mask_end = np.random.randint(236*2, 250*2)
        # index = np.random.randint(80, 90)
        # mask_B[mask_B.shape[1] // 2 - index:mask_end, 30:-30] = 0
        mask_end = np.random.randint(480, 500)
        index = np.random.randint(15, 30)
        mask_B[index:mask_end, 70:-70] = 0
        img = Image.fromarray(img)

        mask_B = Image.fromarray(mask_B)
        img = self.transforms_image(img)
        mask_B = self.transforms_image(mask_B)

        x = np.where((idx >= self.label_starts) * (idx < self.label_ends))[0]

        audio = torch.tensor(feature_3dmm)
        # s3 = time.time()
        # print('get 3dmm and mask ', s3 - s2)
        # 保证real_A_index不是idx
        max_i = 0
        real_A_index = random.randint(self.label_starts[x], self.label_ends[x] - 1)
        while real_A_index == idx:
            max_i += 1
            real_A_index = random.randint(self.label_starts[x], self.label_ends[x] - 1)
            if max_i > 5:
                break

        imgA_path, _ = self.labels[real_A_index]
        imgA = np.array(Image.open(imgA_path).convert('RGB'))
        cut_pad1 = np.random.randint(0, 20)
        cut_pad2 = np.random.randint(0, 20)
        imgA = imgA[cut_pad1:256*2 + cut_pad1, cut_pad2:256*2 + cut_pad2]

        ########椭圆##########
        # mask = np.zeros(imgA.shape, dtype=np.uint8)
        # cv2.ellipse(mask, (imgA.shape[1] // 2, imgA.shape[0] // 2 - 165 - cut_pad1),
        #             (imgA.shape[1] // 2 + 25, imgA.shape[0]), 0, 0, 360, (255, 255, 255), -1)
        # ROI = cv2.bitwise_and(imgA, mask)
        # imgA = Image.fromarray(ROI)
        #############################
        # imgA[:imgA.shape[1] // 2 - 40 - index2, :] = 0
        imgA = Image.fromarray(imgA)
        imgA = self.transforms_image(imgA)
        # s4 = time.time()
        # print('end time reala ', s4 - s3)
        return {'A': imgA, 'A_label': audio, 'B': img, 'B_label': audio, 'mask_B': mask_B}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.labels)


if __name__ == '__main__':
    from options.train_options import TrainOptions

    opt = TrainOptions().parse()
    dataset = Facereala3dmmDataset(opt)
    dataset_size = len(dataset)
    print(dataset_size)
    for i, data in enumerate(dataset):
        print(data)
