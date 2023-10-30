import cv2
import numpy as np
from matplotlib import pyplot as plt
import math

np.random.seed(100)  # NumPy 모듈에 대한 시드 설정
num_iterations, inlier_threshold = 10000, 5.0


def compute_homography(kp1, kp2, matches):
    src_pts = []
    dst_pts = []

    for match in matches:
        src_pt = kp1[match.queryIdx].pt
        dst_pt = kp2[match.trainIdx].pt
        src_pts.append(src_pt)
        dst_pts.append(dst_pt)

    src_pts = np.array(src_pts, dtype=np.float32)
    dst_pts = np.array(dst_pts, dtype=np.float32)

    A = []
    for i in range(4):
        x, y = src_pts[i]
        u, v = dst_pts[i]
        A.append([x, y, 1, 0, 0, 0, -x * u, -y * u, -u])
        A.append([0, 0, 0, x, y, 1, -x * v, -y * v, -v])

    A = np.array(A)
    _, _, V = np.linalg.svd(A)

    H = V[-1].reshape(3, 3)
    H /= H[2, 2]

    return H

def ransac(kp1, kp2, matches, num_iterations, inlier_threshold):
    H = None
    best_inliers = 0

    for _ in range(num_iterations):
        random_matches = np.random.choice(matches, 4, replace=False)
        h = compute_homography(kp1, kp2, random_matches)

        inliers = 0
        for match in matches:
            src_pt = kp1[match.queryIdx].pt
            dst_pt = kp2[match.trainIdx].pt
            src_pt = np.array([src_pt[0], src_pt[1], 1])
            projected_pt = np.dot(h, src_pt)
            projected_pt /= projected_pt[2]

            dist = np.sqrt((projected_pt[0] - dst_pt[0]) ** 2 + (projected_pt[1] - dst_pt[1]) ** 2)

            if dist <= inlier_threshold:
                inliers += 1

        if inliers > best_inliers:
            best_inliers = inliers
            H = h
        
        if _ == 1000:
            print("iteration : ",_, " ",best_inliers)
            print(H)
        
        if _ == 2000:
            print("iteration : ",_, " ",best_inliers)
            print(H)

        if _ == 9999:
            print("iteration : ",_, " ",best_inliers)
            print(H)

    return H


def warp_perspective(image1, image2, H):
    
    pts1 = np.float32([[0, 0], [0, image2.shape[0]], [image2.shape[1], image2.shape[0]], [image2.shape[1], 0]]).reshape(-1, 1, 2)
    pts2 = np.float32([[0, 0], [0, image1.shape[0]], [image1.shape[1], image1.shape[0]], [image1.shape[1], 0]]).reshape(-1, 1, 2)
    pts2_ = cv2.perspectiveTransform(pts2, homographyMat)
    pts = np.concatenate((pts1, pts2_), axis=0)

    [xmin, ymin] = np.int32(pts.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(pts.max(axis=0).ravel() + 0.5)

    Ht = np.array([[1, 0, -xmin],
                   [0, 1, -ymin], 
                   [0, 0, 1]])

    result = cv2.warpPerspective(image1, Ht.dot(homographyMat), (xmax-xmin, ymax-ymin))
    result[-ymin:image2.shape[0]-ymin, -xmin:image2.shape[1]-xmin] = image2
    
    return result

image1 = cv2.imread("image/source_image1.jpg")
image2 = cv2.imread("image/source_image2.jpg")

image1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
image2_gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

orb_detector = cv2.ORB_create()

key_points1, descriptors1 = orb_detector.detectAndCompute(image1_gray, None)
key_points2, descriptors2 = orb_detector.detectAndCompute(image2_gray, None)

keyImage1 = cv2.drawKeypoints(image1_gray, key_points1, np.array([]), (0, 0, 255))
keyImage2 = cv2.drawKeypoints(image2_gray, key_points2, np.array([]), (0, 0, 255))

cv2.imwrite('result/keyImage1.jpg', keyImage1)
cv2.imwrite('result/keyImage2.jpg', keyImage2)


brute_force_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck= True)

matches = brute_force_matcher.match(descriptors1, descriptors2)

matches = sorted(matches, key = lambda x:x.distance)



new_image = cv2.drawMatches(image1, key_points1, image2, key_points2, matches, outImg=None, flags=2)

cv2.imwrite('result/matching_poing.jpg', new_image)


srce_pts = np.float32([ key_points1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
dest_pts = np.float32([ key_points2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

homographyMat = ransac(key_points1, key_points2, matches, num_iterations, inlier_threshold)
print(homographyMat)

result = warp_perspective(image1, image2, homographyMat)

cv2.imwrite('result/result.jpg', result)