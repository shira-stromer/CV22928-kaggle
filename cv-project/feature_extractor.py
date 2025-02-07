import os
import cv2
import torch
import numpy as np
from super_glue.models.matching import Matching
from super_glue.models.utils import read_image, frame2tensor


class FeatureExtractor:
    def __init__(self, images, device, num_features=8000):
        self.images = images
        self.num_features = num_features
        self.detector = cv2.SIFT_create(nfeatures=num_features, contrastThreshold=-10000, edgeThreshold=-10000)
        self.bf_matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        self.config = {
            "superpoint": {
                "nms_radius": 3,
                "keypoint_threshold": 0.005,
                "max_keypoints": 2048
            },
            "superglue": {
                "weights": "outdoor",
                "sinkhorn_iterations": 30,
                "match_threshold": 0.4,
            }
        }

        self.device = device
        self.model = Matching(self.config)
        self.model.to(device)
        self.model.eval()

    def resize_with_uniform_scale(self, img, target_size, common_scale):
        """ Resize an image using a common scale and pad to match target size. """
        h, w = img.shape[:2]
        
        # Compute new dimensions
        new_w, new_h = int(w * common_scale), int(h * common_scale)
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Compute padding
        pad_w = (target_size[0] - new_w) // 2
        pad_h = (target_size[1] - new_h) // 2

        # Pad to match target size exactly
        padded = cv2.copyMakeBorder(resized, pad_h, target_size[1] - new_h - pad_h,
                                    pad_w, target_size[0] - new_w - pad_w,
                                    cv2.BORDER_CONSTANT, value=[0, 0, 0])

        return padded, common_scale

    def resize_image(self, img1_path, img2_path, target_size=[840, 640]):
        # Load images
        img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

        # Original image shapes
        h1, w1 = img1.shape[:2]  # (1046, 686)
        h2, w2 = img2.shape[:2]  # (738, 993)

        # Target size
        target_w, target_h = max(w1, w2), h1 if w1 == max(w1, w2) else h2

        # Compute individual scaling factors
        scale1_x, scale1_y = target_w / w1, target_h / h1
        scale2_x, scale2_y = target_w / w2, target_h / h2

        # Choose a common scale (smallest of all scaling factors)
        common_scale = min(scale1_x, scale1_y, scale2_x, scale2_y)

        # Resize images using the common scale
        img1_resized, scale1 = self.resize_with_uniform_scale(img1, (target_w, target_h), common_scale)
        img2_resized, scale2 = self.resize_with_uniform_scale(img2, (target_w, target_h), common_scale)

        # Ensure scaling factors are equal
        assert np.isclose(scale1, scale2), "Scaling factors must be equal!"
        return frame2tensor(img1_resized, self.device), frame2tensor(img2_resized, self.device), common_scale

    def extract_sift_features(self, image):
        '''Compute SIFT features for a given image.'''
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Note that you may actually get more than num_features features, as a feature for one point can have multiple orientations (this is rare).    
        keypoints, descriptors = self.detector.detectAndCompute(gray, None)
        #print(f"Number of keypoints: {len(keypoints)}")
        #print(f"Descriptors shape: {descriptors.shape}")
        keypoints = keypoints[:self.num_features]
        descriptors = descriptors[:self.num_features]
        return keypoints, descriptors

    def array_from_CvKps(self, kps):
        '''Convenience function to convert OpenCV keypoints into a simple numpy array.'''
        return np.array([kp.pt for kp in kps])

    def compute_features(self, pair_features_path, item):
        if pair_features_path and os.path.exists(pair_features_path):
            return torch.load(pair_features_path)

        try:
            im0, im1 = item['im1'], item['im2']
            img1_path, img2_path = self.images[im0], self.images[im1]

            #image9 = read_image(img1_path, self.device, [840, 640], 0, True)
            #image0, image1, scale = self.resize_image(img1_path, img2_path, [840, 840])

            #with torch.no_grad():
                #self.model.superpoint({'data': frame2tensor(cv2.imread(str(img1_path), cv2.IMREAD_GRAYSCALE))})
            with torch.no_grad():
                output = self.model.forward(
                    {
                        'image0': frame2tensor(cv2.imread(str(img1_path), cv2.IMREAD_GRAYSCALE), device=self.device), 
                        'image1': frame2tensor(cv2.imread(str(img2_path), cv2.IMREAD_GRAYSCALE), device=self.device),
                        'keypoints0_path': os.path.join(f'tmp/images_features/{im0}.pt'),
                        'keypoints1_path': os.path.join(f'tmp/images_features/{im1}.pt')
                     })

            keypoints0, keypoints1 = output["keypoints0"][0].cpu(), output["keypoints1"][0].cpu()
            #descriptors0, descriptors1 = output["descriptors0"][0].cpu(), output["descriptors1"][0].cpu()
 
            matches0 = output["matches0"][0].cpu()

            valid_mask = matches0 != -1              # Mask: [True, False, True, True, False]
            valid_indices = torch.nonzero(valid_mask).squeeze(1)  # Indices of valid matches: [0, 2, 3]

            # Step 2: Filter keypoints0 for valid matches
            keypoints0 = keypoints0[valid_indices].numpy()       # Keep keypoints with matches

            # Step 3: Reorganize keypoints1 based on matched indices
            matched_indices_in_kp1 = matches0[valid_indices]      # [2, 4, 1] → indices in keypoints1
            keypoints1 = keypoints1[matched_indices_in_kp1].numpy() # Select corresponding keypoints
            #mconf = conf[valid]

            features = (keypoints0, keypoints1, None, None)
            # Save the Fundamental matrix

            torch.save(features, pair_features_path)
            return features

        except Exception as e:
            print(f"Failure when computing features for: {item} - {e}")


if __name__ == "__main__":
    images = {'im0': 'tmp/train/taj_mahal/images/02760416_4808737899.jpg', 'im1': 'tmp/train/taj_mahal/images/02760416_4808737899.jpg'}
    fe = FeatureExtractor(images, 'mps', 2048)
    fe.compute_features(None, {'im1': 'im0', 'im2': 'im1'})