import numpy as np
import random
import math
from scipy.spatial.distance import squareform, pdist, cdist
from skimage.util import img_as_float
from skimage import color

### Clustering Methods for 1-D points
def kmeans(features, k, num_iters=500):
    """ Use kmeans algorithm to group features into k clusters.

    K-Means algorithm can be broken down into following steps:
        1. Randomly initialize cluster centers
        2. Assign each point to the closest center
        3. Compute new center of each cluster
        4. Stop if cluster assignments did not change
        5. Go to step 2

    Args:
        features - Array of N features vectors. Each row represents a feature
            vector.
        k - Number of clusters to form.
        num_iters - Maximum number of iterations the algorithm will run.

    Returns:
        assignments - Array representing cluster assignment of each point.
            (e.g. i-th point is assigned to cluster assignments[i])
    """
    print(features.shape)
    N, D = features.shape

    assert N >= k, 'Number of clusters cannot be greater than number of points'

    # Randomly initalize cluster centers
    idxs = np.random.choice(N, size=k, replace=False)
    centers = features[idxs]
    assignments = np.zeros(N, dtype=np.uint32)

    for n in range(num_iters):
        ### YOUR CODE HERE
        pass
        stop_flag = True
        for feature_idx in range(N): # 遍历每一个点
            feature = features[feature_idx] # 当前遍历到的点
            assignment = 0
            center = centers[0]
            distance = np.sum(np.square(feature - center))
            for center_idx in range(1, k): # 寻找距离最小的中心点，将当前的点归为该 cluster
                new_center = centers[center_idx]
                new_distance = np.sum(np.square(feature - new_center))
                if new_distance < distance:
                    assignment = center_idx
                    distance = new_distance
                    center = new_center
            if assignments[feature_idx] != assignment:
                stop_flag = False # cluster assignments 没有变化时才提前结束
                assignments[feature_idx] = assignment

        for center_idx in range(k): # 更新每一个 cluster 的中心
            center_features = [] # 当前 cluster 所管辖的所有点
            for feature_idx in range(N):
                if assignments[feature_idx] == center_idx:
                    center_features.append(features[feature_idx])
            center_features = np.array(center_features)
            if center_features.shape[0] == 0:
                continue
            centers[center_idx, 0] = np.mean(center_features[:, 0])
            centers[center_idx, 1] = np.mean(center_features[:, 1])

        if stop_flag:
            break
        ### END YOUR CODE

    return assignments

### Clustering Methods for colorful image
def kmeans_color(features, k, num_iters=500):
    new_features = features.reshape(-1, 3)
    H, W = features.shape[0], features.shape[1]
    N = H * W
    assignments = np.zeros(N, dtype=np.uint32)
    #Like the kmeans function above
    ### YOUR CODE HERE
    pass
    idxs = np.random.choice(N, size=k, replace=False)
    centers = (new_features[idxs]).astype(np.float32)

    for n in range(num_iters):
        stop_flag = True

        for feature_idx in range(N): # 遍历每一个点
            feature = new_features[feature_idx] # 当前遍历到的点
            assignment = 0
            center = centers[0]
            distance = np.sum(np.square(feature - center))
            for center_idx in range(1, k): # 寻找距离最小的中心点，将当前的点归为该 cluster
                new_center = centers[center_idx]
                new_distance = np.sum(np.square(feature - new_center))
                if new_distance < distance:
                    assignment = center_idx
                    distance = new_distance
                    center = new_center
            if assignments[feature_idx] != assignment:
                stop_flag = False # cluster assignments 没有变化时才提前结束
                assignments[feature_idx] = assignment

        for center_idx in range(k): # 更新每一个 cluster 的中心
            center_features = [] # 当前 cluster 所管辖的所有点
            for feature_idx in range(N):
                if assignments[feature_idx] == center_idx:
                    center_features.append(new_features[feature_idx])
            center_features = np.array(center_features)
            if center_features.shape[0] == 0:
                continue
            centers[center_idx, 0] = np.mean(center_features[:, 0])
            centers[center_idx, 1] = np.mean(center_features[:, 1])
            centers[center_idx, 2] = np.mean(center_features[:, 2])

        if stop_flag:
            break
    
    assignments = assignments.reshape(H, W)
    ### END YOUR CODE

    return assignments





#找每个点最后会收敛到的地方（peak）
def findpeak(data, idx, r):
    t = 0.01
    shift = np.array([1])
    data_point = data[:, idx]
    dataT = data.T
    data_pointT = data_point.T
    data_pointT = data_pointT.reshape(1, 3)

    # Runs until the shift is smaller than the set threshold
    while (shift > t).any():
        ### YOUR CODE HERE
        pass
        # 计算当前点和所有点之间的距离
        distance = np.sqrt(np.sum(np.square(data_pointT - dataT), axis=1, keepdims=True))

        # 筛选出在半径r内的点，计算mean vector（这里是最简单的均值，也可尝试高斯加权）
        filtered_points = dataT[(np.where(distance <= r))[0]]
        mean_vector = np.mean(filtered_points, axis=0)
        shift = np.sqrt(np.sum(np.square(mean_vector - data_pointT)))

        # 用新的center（peak）更新当前点，直到满足要求跳出循环
        data_pointT = mean_vector.reshape((1, 3))
        ### END YOUR CODE

    return data_pointT.T


# Mean shift algorithm
# 可以改写代码，鼓励自己的想法，但请保证输入输出与notebook一致
def meanshift(data, r):
    labels = np.zeros(len(data.T))
    peaks = [] #聚集的类中心
    label_no = 1 #当前label
    labels[0] = label_no

    # findpeak is called for the first index out of the loop
    peak = findpeak(data, 0, r)
    peakT = np.concatenate(peak, axis=0).T
    peaks.append(peakT)

    # Every data point is iterated through
    for idx in range(0, len(data.T)):
        ### YOUR CODE HERE
        pass
        # 遍历数据，寻找当前点的peak
        peak = np.concatenate(findpeak(data, idx, r), axis=0).T

        # 并实时关注当前peak是否会收敛到一个新的聚类（和已有peaks比较）
        new_peak_flag = True
        for i in range(len(peaks)):
            if np.sqrt(np.sum(np.square(peak - peaks[i]))) < r:
                new_peak_flag = False
        if new_peak_flag: # 若是，更新label_no，peaks，labels，继续
            label_no += 1
            peaks.append(peak)
            labels[idx] = label_no
        else: # 若不是，当前点就属于已有类，继续
            for i in range(len(peaks)):
                if np.sqrt(np.sum(np.square(peak - peaks[i]))) < r:
                    labels[idx] = i + 1
        ### END YOUR CODE
    #print(set(labels))
    return labels, np.array(peaks).T


# image segmentation
def segmIm(img, r):
    # Image gets reshaped to a 2D array
    img_reshaped = np.reshape(img, (img.shape[0] * img.shape[1], 3))

    # We will work now with CIELAB images
    imglab = color.rgb2lab(img_reshaped)
    # segmented_image is declared
    segmented_image = np.zeros((img_reshaped.shape[0], img_reshaped.shape[1]))


    labels, peaks = meanshift(imglab.T, r)
    # Labels are reshaped to only one column for easier handling
    labels_reshaped = np.reshape(labels, (labels.shape[0], 1))

    # We iterate through every possible peak and its corresponding label
    for label in range(0, peaks.shape[1]):
        # Obtain indices for the current label in labels array
        inds = np.where(labels_reshaped == label + 1)[0]

        # The segmented image gets indexed peaks for the corresponding label
        corresponding_peak = peaks[:, label]
        segmented_image[inds, :] = corresponding_peak
    # The segmented image gets reshaped and turn back into RGB for display
    segmented_image = np.reshape(segmented_image, (img.shape[0], img.shape[1], 3))

    res_img=color.lab2rgb(segmented_image)
    res_img=color.rgb2gray(res_img)
    return res_img


### Quantitative Evaluation
def compute_accuracy(mask_gt, mask):
    """ Compute the pixel-wise accuracy of a foreground-background segmentation
        given a ground truth segmentation.

    Args:
        mask_gt - The ground truth foreground-background segmentation. A
            logical of size H x W where mask_gt[y, x] is 1 if and only if
            pixel (y, x) of the original image was part of the foreground.
        mask - The estimated foreground-background segmentation. A logical
            array of the same size and format as mask_gt.

    Returns:
        accuracy - The fraction of pixels where mask_gt and mask agree. A
            bigger number is better, where 1.0 indicates a perfect segmentation.
    """

    accuracy = None
    ### YOUR CODE HERE
    pass
    # 计算正反两种情况的准确率，然后取最大值
    total_counts = mask_gt.size
    correct_counts_1 = np.count_nonzero(mask_gt == mask)
    accuracy_1 = correct_counts_1 / total_counts
    correct_counts_2 = np.count_nonzero(mask_gt != mask)
    accuracy_2 = correct_counts_2 / total_counts
    accuracy = max(accuracy_1, accuracy_2)
    ### END YOUR CODE

    return accuracy

