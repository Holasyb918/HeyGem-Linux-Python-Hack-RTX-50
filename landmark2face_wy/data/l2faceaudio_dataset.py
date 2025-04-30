import os.path
import random
from data.base_dataset import BaseDataset, get_params, get_transform
import torchvision.transforms as transforms
from data.image_folder import make_dataset
from PIL import Image, ImageEnhance
import numpy as np
import cv2
import torch


def get_idts(config_name):
    idts = list()
    with open(os.path.join('../config', config_name + '.txt')) as f:
        for line in f:
            line = line.strip()
            idts.append(line)
    return idts


class L2FaceAudioDataset(BaseDataset):
    def __init__(self, opt, mode=None):
        BaseDataset.__init__(self, opt)
        img_size = opt.img_size
        idts = get_idts(opt.name.split('_')[0])
        print("---------load data list--------: ", idts)
        if mode == 'train':
            self.labels = []
            for idt_name in idts:
                # root = '../AnnVI/feature/{}'.format(idt_name)
                root = os.path.join(opt.feature_path, idt_name)
                if opt.audio_feature == "mfcc":
                    training_data_path = os.path.join(root, '{}_{}.t7'.format(img_size, mode))
                else:
                    training_data_path = os.path.join(root, '{}_{}_{}.t7'.format(img_size, mode, opt.audio_feature))
                training_data = torch.load(training_data_path)
                img_paths = training_data['img_paths']
                audio_features = training_data['audio_features']
                index = [i[0].split('/')[-1] for i in img_paths]

                image_dir = '{}/{}_dlib_crop'.format(root, img_size)
                # label_dir = '{}/512_landmark_crop'.format(root)

                # if 'man' in opt.name:
                #     imgs.sort(key=lambda x:int(x.split('.')[0]))
                # else:
                #     imgs.sort(key=lambda x: (int(x.split('.')[0].split('-')[0]), int(x.split('.')[0].split('-')[1])))
                for img in range(len(index)):
                    img_path = os.path.join(image_dir, index[img])
                    audio_feature = audio_features[img]
                    self.labels.append([img_path, audio_feature])
            # transforms.Resize([img_size, img_size], Image.BICUBIC),
            self.transforms_image = transforms.Compose([transforms.ToTensor(),
                                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            # transforms.Resize([img_size, img_size], Image.BICUBIC),
            self.transforms_label = transforms.Compose([transforms.ToTensor(),
                                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            self.shuffle()
        elif mode == 'test':
            self.labels = []
            for idt_name in idts:
                # root = '../AnnVI/feature/{}'.format(idt_name)
                root = os.path.join(opt.feature_path, idt_name)
                if opt.audio_feature == "mfcc":
                    training_data_path = os.path.join(root, '{}_{}.t7'.format(img_size, mode))
                else:
                    training_data_path = os.path.join(root, '{}_{}_{}.t7'.format(img_size, mode, opt.audio_feature))
                training_data = torch.load(training_data_path)
                img_paths = training_data['img_paths']
                audio_features = training_data['audio_features']
                index = [i[0].split('/')[-1] for i in img_paths]

                image_dir = '{}/{}_dlib_crop'.format(root, img_size)
                # label_dir = '{}/512_landmark_crop'.format(root)

                # if 'man' in opt.name:
                #     imgs.sort(key=lambda x:int(x.split('.')[0]))
                # else:
                #     imgs.sort(key=lambda x: (int(x.split('.')[0].split('-')[0]), int(x.split('.')[0].split('-')[1])))
                for img in range(len(index)):
                    img_path = os.path.join(image_dir, index[img])
                    audio_feature = audio_features[img]
                    self.labels.append([img_path, audio_feature])
                # transforms.Resize([img_size, img_size], Image.BICUBIC),
            self.transforms_image = transforms.Compose([transforms.ToTensor(),
                                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            # transforms.Resize([img_size, img_size], Image.BICUBIC),
            self.transforms_label = transforms.Compose([transforms.ToTensor(),
                                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            self.shuffle()

    def shuffle(self):
        random.shuffle(self.labels)

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
        cv2.setNumThreads(0)
        img_path, audio_feature = self.labels[index]
        img = np.array(Image.open(img_path).convert('RGB'))
        img = np.array(np.clip(img + np.random.randint(-20, 20, size=3, dtype='int8'), 0, 255), dtype='uint8')
        cut_pad1 = np.random.randint(0, 10)
        cut_pad2 = np.random.randint(0, 10)
        img = img[cut_pad1:256 + cut_pad1, cut_pad2:256 + cut_pad2]

        ####mask遮眼睛#####
        mask = np.ones(img.shape, dtype=np.uint8) * 255
        mask[20 - cut_pad1:70 - cut_pad1, 55 - cut_pad2:-55 - cut_pad2] = 0
        img = cv2.bitwise_and(img, mask)

        mask_B = img.copy()
        mask_end = np.random.randint(236, 256)
        ##########之前老的矩形mask#############
        mask_B[mask_B.shape[1] // 2 - np.random.randint(40, 50):mask_end, 30:-30] = 0
        ##########之前老的矩形mask#############
        ##########蔡星宇三角mask#############
        # mask_B = self.add_mouth_mask2(mask_B)
        ##########蔡星宇三角mask#############
        # mask_B[mask_B.shape[1] // 2 - 50:, 30:-30] = 0
        img = Image.fromarray(img)
        mask_B = Image.fromarray(mask_B)
        img = self.transforms_image(img)
        mask_B = self.transforms_image(mask_B)
        # lab = Image.open(lab_path).convert('RGB')
        # lab = self.transforms_label(lab)
        audio = np.zeros((256, 256), dtype=np.float32)
        audio_feature = np.array(audio_feature)
        audio[:audio_feature.shape[0], :audio_feature.shape[1]] = audio_feature
        audio = torch.tensor([audio])

        imgA_path, _ = random.sample(self.labels, 1)[0]
        imgA = np.array(Image.open(imgA_path).convert('RGB'))
        cut_pad1 = np.random.randint(0, 10)
        cut_pad2 = np.random.randint(0, 10)
        imgA = imgA[cut_pad1:256 + cut_pad1, cut_pad2:256 + cut_pad2]
        imgA = cv2.bitwise_and(imgA, mask)
        imgA = Image.fromarray(imgA)
        imgA = self.transforms_image(imgA)
        return {'A': imgA, 'A_label': audio, 'B': img, 'B_label': audio, 'mask_B': mask_B}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.labels)


if __name__ == '__main__':
    from options.train_options import TrainOptions

    opt = TrainOptions().parse()
    dataset = L2FaceDataset(opt)
    dataset_size = len(dataset)
    print(dataset_size)
    for i, data in enumerate(dataset):
        print(data)