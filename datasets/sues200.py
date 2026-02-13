import os
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
import copy
import random
import time
from tqdm import tqdm


def get_sues200_data(satellite_folder, drone_folder, drone_height, class_ids):
    """
    Get SUES200 data structure
    Args:
        satellite_folder: path to satellite images
        drone_folder: path to drone images  
        drone_height: height folder to use (150, 200, 250, 300)
        class_ids: list of class IDs to include
    """
    data = {}
    
    for class_id in class_ids:
        data[class_id] = {}
        
        # Satellite data
        sat_path = os.path.join(satellite_folder, class_id)
        if os.path.exists(sat_path):
            sat_files = [f for f in os.listdir(sat_path) if f.endswith(('.jpg', '.png'))]
            data[class_id]['satellite'] = {
                'path': sat_path,
                'files': sat_files
            }
        
        # Drone data
        drone_path = os.path.join(drone_folder, class_id, str(drone_height))
        if os.path.exists(drone_path):
            drone_files = [f for f in os.listdir(drone_path) if f.endswith(('.jpg', '.png'))]
            data[class_id]['drone'] = {
                'path': drone_path,
                'files': drone_files
            }
    
    # Only keep class IDs that have both satellite and drone data
    valid_data = {}
    for class_id in data:
        if 'satellite' in data[class_id] and 'drone' in data[class_id]:
            if len(data[class_id]['satellite']['files']) > 0 and len(data[class_id]['drone']['files']) > 0:
                valid_data[class_id] = data[class_id]
    
    return valid_data


class SUES200DatasetTrain(Dataset):

    def __init__(self,
                 satellite_folder,
                 drone_folder,
                 drone_height,
                 class_ids,
                 transforms_query=None,
                 transforms_gallery=None,
                 prob_flip=0.5,
                 shuffle_batch_size=128):
        super().__init__()

        self.data_dict = get_sues200_data(satellite_folder, drone_folder, drone_height, class_ids)
        self.ids = list(self.data_dict.keys())
        self.ids.sort()

        self.pairs = []

        for idx in self.ids:
            # For satellite images (query) - 取排序后的第一张卫星图
            sat_files = sorted(self.data_dict[idx]['satellite']['files'])
            sat_path = self.data_dict[idx]['satellite']['path']
            
            # For drone images (gallery)
            drone_files = sorted(self.data_dict[idx]['drone']['files'])
            drone_path = self.data_dict[idx]['drone']['path']

            # Create pairs: 每个class取第一张satellite图，与所有drone图配对
            # 根据数据集结构，每类只有1张卫星图，所以取第一张即可
            if len(sat_files) > 0:
                sat_img_path = os.path.join(sat_path, sat_files[0])
                for drone_file in drone_files:
                    drone_img_path = os.path.join(drone_path, drone_file)
                    self.pairs.append((idx, sat_img_path, drone_img_path))

        self.transforms_query = transforms_query
        self.transforms_gallery = transforms_gallery
        self.prob_flip = prob_flip
        self.shuffle_batch_size = shuffle_batch_size

        self.samples = copy.deepcopy(self.pairs)

    def __getitem__(self, index):

        idx, query_img_path, gallery_img_path = self.samples[index]

        # Load satellite image (query)
        query_img = cv2.imread(query_img_path)
        query_img = cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB)

        # Load drone image (gallery)
        gallery_img = cv2.imread(gallery_img_path)
        gallery_img = cv2.cvtColor(gallery_img, cv2.COLOR_BGR2RGB)

        if np.random.random() < self.prob_flip:
            query_img = cv2.flip(query_img, 1)
            gallery_img = cv2.flip(gallery_img, 1)

        # Apply transforms
        if self.transforms_query is not None:
            query_img = self.transforms_query(image=query_img)['image']

        if self.transforms_gallery is not None:
            gallery_img = self.transforms_gallery(image=gallery_img)['image']

        return query_img, gallery_img, idx

    def __len__(self):
        return len(self.samples)

    def shuffle(self):
        '''
        Custom shuffle function for unique class_id sampling in batch
        '''
        print("\nShuffle Dataset:")

        pair_pool = copy.deepcopy(self.pairs)
        random.shuffle(pair_pool)

        pairs_epoch = set()
        idx_batch = set()

        batches = []
        current_batch = []
        break_counter = 0

        pbar = tqdm()

        while True:
            pbar.update()

            if len(pair_pool) > 0:
                pair = pair_pool.pop(0)
                idx, _, _ = pair

                if idx not in idx_batch and pair not in pairs_epoch:
                    idx_batch.add(idx)
                    current_batch.append(pair)
                    pairs_epoch.add(pair)
                    break_counter = 0
                else:
                    if pair not in pairs_epoch:
                        pair_pool.append(pair)
                    break_counter += 1

                if break_counter >= 512:
                    break
            else:
                break

            if len(current_batch) >= self.shuffle_batch_size:
                batches.extend(current_batch)
                idx_batch = set()
                current_batch = []

        pbar.close()
        time.sleep(0.3)

        self.samples = batches

        print("Original Length: {} - Length after Shuffle: {}".format(len(self.pairs), len(self.samples)))
        print("Break Counter:", break_counter)
        print("Pairs left out of last batch to avoid creating noise:", len(self.pairs) - len(self.samples))
        if len(self.samples) > 0:
            print("First Element ID: {} - Last Element ID: {}".format(self.samples[0][0], self.samples[-1][0]))


class SUES200DatasetEval(Dataset):

    def __init__(self,
                 data_folder,
                 mode,
                 drone_height=None,
                 class_ids=None,
                 transforms=None,
                 sample_ids=None,
                 gallery_n=-1):
        super().__init__()

        self.mode = mode
        self.drone_height = drone_height
        self.transforms = transforms
        self.given_sample_ids = sample_ids
        self.gallery_n = gallery_n

        self.images = []
        self.sample_ids = []

        if class_ids is None:
            class_ids = os.listdir(data_folder)

        for class_id in class_ids:
            class_path = os.path.join(data_folder, class_id)
            
            if not os.path.exists(class_path):
                continue
                
            if drone_height is not None:
                # This is drone data, need to go into height subfolder
                height_path = os.path.join(class_path, str(drone_height))
                if not os.path.exists(height_path):
                    continue
                files = [f for f in os.listdir(height_path) if f.endswith(('.jpg', '.png'))]
                for file in files:
                    self.images.append(os.path.join(height_path, file))
                    self.sample_ids.append(class_id)
            else:
                # This is satellite data, files are directly in class folder
                files = [f for f in os.listdir(class_path) if f.endswith(('.jpg', '.png'))]
                for file in files:
                    self.images.append(os.path.join(class_path, file))
                    self.sample_ids.append(class_id)

    def __getitem__(self, index):
        img_path = self.images[index]
        sample_id = self.sample_ids[index]

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Apply transforms
        if self.transforms is not None:
            img = self.transforms(image=img)['image']

        label = int(sample_id)
        if self.given_sample_ids is not None:
            if sample_id not in self.given_sample_ids:
                label = -1

        return img, label

    def __len__(self):
        return len(self.images)

    def get_sample_ids(self):
        return set(self.sample_ids)


def get_transforms(img_size,
                   mean=[0.485, 0.456, 0.406],
                   std=[0.229, 0.224, 0.225]):
    val_transforms = A.Compose([A.Resize(img_size[0], img_size[1], interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),
                                A.Normalize(mean, std),
                                ToTensorV2(),
                                ])

    train_sat_transforms = A.Compose([A.ImageCompression(quality_lower=90, quality_upper=100, p=0.5),
                                      A.Resize(img_size[0], img_size[1], interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),
                                      A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.15,
                                                    always_apply=False, p=0.5),
                                      A.OneOf([
                                          A.AdvancedBlur(p=1.0),
                                          A.Sharpen(p=1.0),
                                      ], p=0.3),
                                      A.OneOf([
                                          A.GridDropout(ratio=0.4, p=1.0),
                                          A.CoarseDropout(max_holes=25,
                                                          max_height=int(0.2 * img_size[0]),
                                                          max_width=int(0.2 * img_size[0]),
                                                          min_holes=10,
                                                          min_height=int(0.1 * img_size[0]),
                                                          min_width=int(0.1 * img_size[0]),
                                                          p=1.0),
                                      ], p=0.3),
                                      A.RandomRotate90(p=1.0),
                                      A.Normalize(mean, std),
                                      ToTensorV2(),
                                      ])

    train_drone_transforms = A.Compose([A.ImageCompression(quality_lower=90, quality_upper=100, p=0.5),
                                        A.Resize(img_size[0], img_size[1], interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),
                                        A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.15,
                                                      always_apply=False, p=0.5),
                                        A.OneOf([
                                            A.AdvancedBlur(p=1.0),
                                            A.Sharpen(p=1.0),
                                        ], p=0.3),
                                        A.OneOf([
                                            A.GridDropout(ratio=0.4, p=1.0),
                                            A.CoarseDropout(max_holes=25,
                                                            max_height=int(0.2 * img_size[0]),
                                                            max_width=int(0.2 * img_size[0]),
                                                            min_holes=10,
                                                            min_height=int(0.1 * img_size[0]),
                                                            min_width=int(0.1 * img_size[0]),
                                                            p=1.0),
                                        ], p=0.3),
                                        A.Normalize(mean, std),
                                        ToTensorV2(),
                                        ])

    return val_transforms, train_sat_transforms, train_drone_transforms 