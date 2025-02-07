"""
This script processes image pairs for fundamental and essential matrix estimation,
and evaluates the accuracy of predicted transformations.

Dependencies:
- NumPy for numerical computations
- OpenCV for computer vision tasks
- PyTorch for tensor operations and model handling
"""

from dataset import VisualDataset
import torch
import tqdm
import numpy as np
from swifter import swifter
import os

np.random.seed(22)
torch.manual_seed(22)
import cv2


eps = 1e-15

def NormalizeKeypoints(keypoints, K):
    """Normalizes keypoints using the camera intrinsic matrix K."""
    C_x = K[0, 2]
    C_y = K[1, 2]
    f_x = K[0, 0]
    f_y = K[1, 1]
    keypoints = (keypoints - np.array([[C_x, C_y]])) / np.array([[f_x, f_y]])
    return keypoints

def ComputeEssentialMatrix(F, K1, K2, kp1, kp2):
    '''Compute the Essential matrix from the Fundamental matrix, given the calibration matrices. Note that we ask participants to estimate F, i.e., without relying on known intrinsics.'''
    
    # Warning! Old versions of OpenCV's RANSAC could return multiple F matrices, encoded as a single matrix size 6x3 or 9x3, rather than 3x3.
    # We do not account for this here, as the modern RANSACs do not do this:
    # https://opencv.org/evaluating-opencvs-new-ransacs
    assert F.shape[0] == 3, 'Malformed F?'

    # Use OpenCV's recoverPose to solve the cheirality check:
    # https://docs.opencv.org/4.5.4/d9/d0c/group__calib3d.html#gadb7d2dfcc184c1d2f496d8639f4371c0
    E = np.matmul(np.matmul(K2.T, F), K1).astype(np.float64)
    
    kp1n = NormalizeKeypoints(kp1, K1)
    kp2n = NormalizeKeypoints(kp2, K2)
    num_inliers, R, T, mask = cv2.recoverPose(E, kp1n, kp2n)

    return E, R, T


def QuaternionFromMatrix(matrix):
    '''Transform a rotation matrix into a quaternion.'''

    M = np.array(matrix, dtype=np.float64, copy=False)[:4, :4]
    m00 = M[0, 0]
    m01 = M[0, 1]
    m02 = M[0, 2]
    m10 = M[1, 0]
    m11 = M[1, 1]
    m12 = M[1, 2]
    m20 = M[2, 0]
    m21 = M[2, 1]
    m22 = M[2, 2]

    K = np.array([[m00 - m11 - m22, 0.0, 0.0, 0.0],
              [m01 + m10, m11 - m00 - m22, 0.0, 0.0],
              [m02 + m20, m12 + m21, m22 - m00 - m11, 0.0],
              [m21 - m12, m02 - m20, m10 - m01, m00 + m11 + m22]])
    K /= 3.0

    # The quaternion is the eigenvector of K that corresponds to the largest eigenvalue.
    w, V = np.linalg.eigh(K)
    q = V[[3, 0, 1, 2], np.argmax(w)]

    if q[0] < 0:
        np.negative(q, q)

    return q

def ComputeErrorForOneExample(q_gt, T_gt, q, T, scale):
    '''Compute the error metric for a single example.
    
    The function returns two errors, over rotation and translation. These are combined at different thresholds by ComputeMaa in order to compute the mean Average Accuracy.'''
    
    q_gt_norm = q_gt / (np.linalg.norm(q_gt) + eps)
    q_norm = q / (np.linalg.norm(q) + eps)

    loss_q = np.maximum(eps, (1.0 - np.sum(q_norm * q_gt_norm)**2))
    err_q = np.arccos(1 - 2 * loss_q)

    # Apply the scaling factor for this scene.
    T_gt_scaled = T_gt * scale
    T_scaled = T * np.linalg.norm(T_gt) * scale / (np.linalg.norm(T) + eps)

    err_t = min(np.linalg.norm(T_gt_scaled - T_scaled), np.linalg.norm(T_gt_scaled + T_scaled))

    return err_q * 180 / np.pi, err_t

def ComputeMaa(err_q, err_t, thresholds_q, thresholds_t):
    '''Compute the mean Average Accuracy at different tresholds, for one scene.'''
    
    assert len(err_q) == len(err_t)
    
    acc, acc_q, acc_t = [], [], []
    for th_q, th_t in zip(thresholds_q, thresholds_t):
        acc += [(np.bitwise_and(np.array(err_q) < th_q, np.array(err_t) < th_t)).sum() / len(err_q)]
        acc_q += [(np.array(err_q) < th_q).sum() / len(err_q)]
        acc_t += [(np.array(err_t) < th_t).sum() / len(err_t)]
    return np.mean(acc), np.array(acc), np.array(acc_q), np.array(acc_t)

def compute_F(dataset, im0, im1, method, confidence, maxIters, reprojThreshold):
    """Estimates the Fundamental matrix between two images using RANSAC-based methods."""
    features_file_path = os.path.join(dataset.featuers_dir, f"{im0}-{im1}.pt")
    item = {'im1':im0, 'im2':im1 }

    (keypoints0, keypoints1, _, _) = dataset.feature_extractor.compute_features(features_file_path, item)
    
    F, inlier_mask = cv2.findFundamentalMat(keypoints0, keypoints1, method, ransacReprojThreshold=reprojThreshold, confidence=confidence, maxIters=maxIters)

    return F, inlier_mask, keypoints0, keypoints1

def FlattenMatrix(M, num_digits=8):
    '''Convenience function to write CSV files.'''
    return ' '.join([f'{v:.{num_digits}e}' for v in M.flatten()])

def predict_row_v1(test_dataset, row, mean_value):
    try:
        F, _, _, _ = compute_F(test_dataset, row['image_1_id'], row['image_2_id'], cv2.USAC_MAGSAC, 0.99, 2000, 0.5)
        if F is None:
            raise Exception('NONE f')
    except Exception as e:
        print(f"Failed to predit F {e}, setting mean value")
        F = mean_value
    return FlattenMatrix(F)
    
def test_parallel(test_dataset, prediction_func, submission_file='submission.csv'):
    df = test_dataset.test_df.copy()
    print(df.columns)
    df['F'] = df.swifter.set_npartitions(npartitions=5).apply(prediction_func, axis=1)
    with open(submission_file, 'w') as f:
        f.write('sample_id,fundamental_matrix\n')
        for sample_id, row in tqdm.tqdm(df.iterrows(), total=len(df), desc="Testing samples"):
            sample_id = row['sample_id']
            f.write(f"{sample_id},{row['F']}\n")

num_features = 2048
max_pairs_per_scene = 5
pair_covis_required = 0.2
device = 'mps'
dataset = VisualDataset(max_pairs_per_scene=max_pairs_per_scene, pair_covis_required=pair_covis_required, num_features=num_features, device=device)
print(len(dataset))

print(f'Scaling factors: {dataset.scaling_factors}')
print()

scaling_dict = dataset.scaling_factors
show_images = True 
num_show_images = 1
verbose = True

# We use two different sets of thresholds over rotation and translation. Do not change this -- these are the values used by the scoring back-end.
thresholds_q = np.linspace(1, 10, 10)
thresholds_t = np.geomspace(0.2, 5, 10)

# Save the per-sample errors and the accumulated metric to dictionaries, for later inspection.
errors = {scene: {} for scene in scaling_dict.keys()}
mAA = {scene: {} for scene in scaling_dict.keys()}


for scene in scaling_dict.keys():
    # Load all pairs, find those with a co-visibility over 0.1, and subsample them.
    covisibility_dict = dataset.scene_pair_covis[scene]
    pairs = list(covisibility_dict['pair'].values)
    
    print(f'-- Processing scene "{scene}": found {len(pairs)} pairs (will keep {min(len(pairs), max_pairs_per_scene)})', flush=True)
    
    # Load ground truth data.
    calib_dict = dataset.scene_cal[scene]

    # Load images and extract SIFT features.
    images_dict = {}
    kp_dict = {}
    desc_dict = {}

    # Process the pairs.
    for pair in tqdm.tqdm(pairs, total=len(pairs)):
        try: 
            id1, id2 = pair.split('-')

            F, inlier_mask, keypoints0, keypoints1 = compute_F(dataset, id1, id2, cv2.USAC_MAGSAC, 0.99, 2000, 0.5)

            inlier_mask = inlier_mask.astype(bool).flatten()   

            assert inlier_mask.sum() > 0, 'inlier_mask.any()'
            inlier_kp_1 = keypoints0[inlier_mask]
            inlier_kp_2 = keypoints1[inlier_mask]

            pred_filepath = os.path.join(dataset.extract_dir, 'pairs_cv2pred', f"{pair}.pt")
            torch.save(F, pred_filepath)
            # Compute the essential matrix.
            fm = dataset.pair_covis[dataset.pair_covis['pair'] == pair]['fundamental_matrix'].iloc[0]
            fundamental_matrix = torch.tensor([float(x) for x in fm.split(' ')], dtype=torch.float32).reshape([3,3])
            E, R, T = ComputeEssentialMatrix(F, calib_dict[id1].K.numpy(), calib_dict[id2].K.numpy(), inlier_kp_1, inlier_kp_2)
            q = QuaternionFromMatrix(R)
            T = T.flatten()

            # Get the relative rotation and translation between these two cameras, given their R and T in the global reference frame.
            R1_gt, T1_gt = calib_dict[id1].R.numpy(), calib_dict[id1].T.numpy().reshape((3, 1))
            R2_gt, T2_gt = calib_dict[id2].R.numpy(), calib_dict[id2].T.numpy().reshape((3, 1))
            dR_gt = np.dot(R2_gt, R1_gt.T)
            dT_gt = (T2_gt - np.dot(dR_gt, T1_gt)).flatten()
            q_gt = QuaternionFromMatrix(dR_gt)
            q_gt = q_gt / (np.linalg.norm(q_gt) + eps)

            # Compute the error for this example.
            err_q, err_t = ComputeErrorForOneExample(q_gt, dT_gt, q, T, scaling_dict[scene])
            errors[scene][pair] = [err_q, err_t]

        except Exception as e:
            print(f"Failed to work on pair: {pair}, {e}")
    
    mAA[scene] = ComputeMaa([v[0] for v in errors[scene].values()], [v[1] for v in errors[scene].values()], thresholds_q, thresholds_t)
    print()
    print(f'Mean average Accuracy on "{scene}": {mAA[scene][0]:.05f}')
    print()


    print()
    print('------- SUMMARY -------')
    print()
    for scene in scaling_dict.keys():
        print(f'-- Mean average Accuracy on "{scene}": {mAA[scene][0]:.05f}')
    print()
    maa = np.mean([mAA[scene][0] for scene in mAA])
    print(f'Mean average Accuracy on dataset: {maa:.05f}')

model_path = "model.pkl"
torch.save(dataset.feature_extractor.model, model_path)

print('Start testing...')
test_dataset = VisualDataset(split='test_images', max_pairs_per_scene=-1, num_features=dataset.num_features, device=device)
mean_value = np.array([x for x in dataset.pair_covis[dataset.pair_covis['covisibility'] <= 0.1]['fundamental_matrix'].apply(lambda x: np.array([float(y) for y in x.split(' ')])).values]).mean(axis=0)
print(mean_value)
prediction_func = lambda row: predict_row_v1(test_dataset, row, mean_value=mean_value)
test_parallel(test_dataset, prediction_func, submission_file='submission__.csv')