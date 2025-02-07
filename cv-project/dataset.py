import os
from torch.utils.data import Dataset
from collections import defaultdict
import zipfile
import pandas as pd
import random
from PIL import Image
import csv
from torchvision.transforms.functional import pil_to_tensor
from torch.utils.data import random_split, DataLoader
from torchvision import transforms
import torch
import cv2
from feature_extractor import FeatureExtractor
import numpy as np
from collections import namedtuple
np.random.seed(22)
torch.manual_seed(22)


Gt = namedtuple('Gt', ['K', 'R', 'T'])

IMAGES_ZIP_PATH = 'cv-22928-2025-a-project.zip'
BLOCKED_SAMPLES = [
    '95733039_5114469579-10193054_9174141765'
    ]

class VisualDataset(Dataset):
    def __init__(self, image_zip_path=IMAGES_ZIP_PATH, split='train', extract_dir='tmp', max_pairs_per_scene=50, pair_covis_required=0.7, num_features=1048, return_images=False, device='mps'):
        self.image_zip_path = image_zip_path
        self.extract_dir = extract_dir
        self.image_dir = os.path.join(extract_dir, split)
        self.images:dict = None
        self.scene_image:defaultdict = None
        self.image_ids:list = None
        self.image_scene:defaultdict = None
        self.scene_cal:dict = {}
        self.scene_pair_covis:dict = {}
        self.pair_covis = {}
        self.featuers_dir = os.path.join(extract_dir, 'features')
        self.feature_extractor = None
        self.max_pairs_per_scene = max_pairs_per_scene    
        self.pair_covis_required = pair_covis_required
        self.num_features = num_features
        self.split = split
        self.return_images = return_images
        self.device = device

        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self._unzip_files()
        self._list_images()
        if self.split == "train":
            self.scaling_factors = pd.read_csv(os.path.join(self.extract_dir, split, 'scaling_factors.csv'), index_col=0).to_dict(orient='records')
            self.scaling_factors = {x['scene']:x['scaling_factor'] for x in self.scaling_factors}
        else:
            self.test_df = pd.read_csv(os.path.join(self.extract_dir, "test.csv"))
        
    def _unzip_files(self):
        """Unzips the XML and image files into the extraction directory."""
        if os.path.exists(self.extract_dir):
            print('Data is already extracted, skipping unzip_files')
        else:
            with zipfile.ZipFile(self.image_zip_path) as zip_ref:
                zip_ref.extractall(self.extract_dir)

        if not os.path.exists(self.featuers_dir):
            os.makedirs(self.featuers_dir)

    def load_calibration(self, filename):
        '''Load calibration data (ground truth) from the csv file.'''
        
        calib_dict = {}
        with open(filename, 'r') as f:
            reader = csv.reader(f, delimiter=',')
            for i, row in enumerate(reader):
                # Skip header.
                if i == 0:
                    continue

                camera_id = row[1]
                K = torch.Tensor([float(v) for v in row[2].split(' ')]).reshape([3, 3])
                R = torch.Tensor([float(v) for v in row[3].split(' ')]).reshape([3, 3])
                T = torch.Tensor([float(v) for v in row[4].split(' ')])
                calib_dict[camera_id] = Gt(K=K, R=R, T=T)
        
        return calib_dict
    
    def _read_covisibility_data(self, filename):
        covisibility_dict = {}
        with open(filename) as f:
            reader = csv.reader(f, delimiter=',')
            for i, row in enumerate(reader):
                # Skip header.
                if i == 0:
                    continue
                covisibility_dict[row[1]] = float(row[2]) # the 1st column is the df index

        return covisibility_dict

    def _list_images(self):
        data_path = self.image_dir
        scenes = []

        self.images = {}
        self.scene_image = defaultdict(list)
        self.image_scene = defaultdict(str)

        for _, dirs, _ in os.walk(data_path):
            scenes = dirs
            break

        for scene in scenes:
            dir_path = os.path.join(data_path, scene)
            if self.split == "train":
                dir_path = os.path.join(dir_path, 'images')
            cal_csv_path = os.path.join(data_path, scene, 'calibration.csv')
            pair_covis_csv_path = os.path.join(data_path, scene, 'pair_covisibility.csv')

            if os.path.exists(cal_csv_path):
                self.scene_cal[scene] = self.load_calibration(cal_csv_path)
                scene_covis_pairs_df = pd.read_csv(pair_covis_csv_path, index_col=0)
                scene_covis_pairs_df = scene_covis_pairs_df[scene_covis_pairs_df['covisibility'] >= self.pair_covis_required]
                scene_covis_pairs_df = scene_covis_pairs_df.sample(frac=1).reset_index(drop=True)
                scene_covis_pairs_df = scene_covis_pairs_df[~scene_covis_pairs_df['pair'].isin(BLOCKED_SAMPLES)]
                scene_covis_pairs_df = scene_covis_pairs_df[:self.max_pairs_per_scene]
                scene_covis_pairs_df['scene'] = scene
                self.scene_pair_covis[scene] = scene_covis_pairs_df

            for file in os.listdir(dir_path):
                if file.endswith('jpg'):
                    file_path = os.path.join(dir_path, file)
                    image_id = file.replace('.jpg', '')
                    self.images[image_id] = file_path
                    self.scene_image[scene].append(image_id)
                    self.image_scene[image_id] = scene

        self.image_ids = list(self.images.keys())

        if self.split == 'train':
            self.pair_covis = pd.concat([v for _, v in self.scene_pair_covis.items()])
            ''' Remove pair where covisibility < 0.1'''
            self.pair_covis = self.pair_covis[~self.pair_covis['pair'].isin(BLOCKED_SAMPLES)]
            self.pair_covis = self.pair_covis.sample(frac=1).reset_index(drop=True)
            random.shuffle(self.image_ids)
        self.feature_extractor = FeatureExtractor(self.images, self.device, self.num_features)

    def __len__(self):
        return len(self.pair_covis)
    
    def len_by_scene(self):
        return {k:len(v) for k,v in self.scene_image.items()}

    def load_image(self, image_id):
        image_path = self.images[image_id]
        return Image.open(image_path).convert("RGB")
    
    def display_pair(self, img1, img2):
        image1, image2 = self.load_image(img1), self.load_image(img2)
        total_width = image1.width + image2.width

        # Create a new blank image to hold the combined images
        combined_image = Image.new('RGB', (total_width, max(image1.height, image2.height)))

        # Paste the images onto the combined image
        combined_image.paste(image1, (0, 0))
        combined_image.paste(image2, (image1.width, 0))

        # Display the combined image
        combined_image.show()

    def compute_features(self, pair_features_path, item):
        if os.path.exists(pair_features_path):
            return torch.load(pair_features_path)

        try:
            im1, im2 = item['im1'], item['im2']
            img1_path, img2_path = self.images[im1], self.images[im2]
            img1 = cv2.imread(img1_path, cv2.COLOR_BGR2RGB)
            img2 = cv2.imread(img2_path, cv2.COLOR_BGR2RGB)

            # Extract and match features
            sift = cv2.SIFT_create()
            keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
            keypoints2, descriptors2 = sift.detectAndCompute(img2, None)

            # Match descriptors
            bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
            matches = bf.knnMatch(descriptors1, descriptors2, k=2)

            # Apply Lowe's ratio test
            good_matches = []
            for m, n in matches:
                if m.distance < 0.8 * n.distance:
                    good_matches.append(m)

            # Get matched keypoint coordinates
            points1 = np.float32([keypoints1[m.queryIdx].pt for m in good_matches])
            points2 = np.float32([keypoints2[m.trainIdx].pt for m in good_matches])

            # Stack matched points
            #matched_points = np.hstack((points1, points2))  # Shape: (num_matches, 4)

            F, mask = cv2.findFundamentalMat(points1, points2, cv2.FM_RANSAC)
            if F is None or F.shape != (3, 3):
                raise ValueError("Fundamental matrix estimation failed.")

            # Filter inliers based on the RANSAC mask
            inliers1 = points1[mask.ravel() == 1]
            inliers2 = points2[mask.ravel() == 1]

            F = torch.tensor(F, dtype=torch.float32)
            torch.save(F, pair_features_path)
            return F
        
        except Exception as e:
            print(f"failure when computing features for : {item} - {e}")

    def __getitem__(self, idx):
        item = self.pair_covis.iloc[idx].to_dict()
        pair = item['pair']
        if self.return_images:
            im1, im2 = item['im1'], item['im2']
            fundamental_matrix = torch.tensor([float(x) for x in item['fundamental_matrix'].split(' ')], dtype=torch.float32).reshape([3,3])
            return self.transform(self.load_image(im1)), self.transform(self.load_image(im2)), fundamental_matrix
            #im1, im2 =  pil_to_tensor(self.load_image(im1).resize([512, 512])), pil_to_tensor(self.load_image(im2).resize([512, 512])), fundamental_matrix
            
        
        features_file_path = os.path.join(self.featuers_dir, f"{pair}.pt")
        F = self.feature_extractor.compute_features(features_file_path, item)
        fundamental_matrix = torch.tensor([float(x) for x in item['fundamental_matrix'].split(' ')], dtype=torch.float32).reshape([3,3])

        return F, fundamental_matrix.numpy()
        # image1, image2 = self.load_image(item['im1']), self.load_image(item['im2'])
        
        # #image1, image2 = pil_to_tensor(image1), pil_to_tensor(image2)
        # image1, image2 = self.transform(image1), self.transform(image2)
        # f_matrix = torch.tensor([float(x) for x in item['fundamental_matrix'].split(' ')]).reshape([3,3])
        # return image1, image2, f_matrix
    
    def getitem_by_pair(self, pair):
        index = self.pair_covis[self.pair_covis['pair'] == pair].index.tolist()
        assert len(index) == 1
        return self[index[0]]

if __name__ == "__main__":
    dataset = VisualDataset()
    for i in range(0, len(dataset)):
        _ = dataset[i]
            




