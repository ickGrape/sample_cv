## このプログラムは、スマホなどで手振れしてしまった画像を固定カメラで取ったような映像にするプログラムです
## 

import cv2
import os
import sys
import shutil
import numpy as np

movie_path = r"C:\myapps\sample-videos-master\movie_data\station_escalator.MOV"



# AKAZEを使って特徴点を抽出　frames　：[frame1, frame2]
# pt1　：特徴量を求める開始座標　例：原点　(0, 0)
# pt2　：特徴量を求める終了座標　例：画像いっぱい　(img.shape[1], img.shape[0])
def get_features(frames, pt1=(0, 0), pt2=None):
    # 特徴量抽出する(AKAZE　：商用利用可能）

    detector = cv2.AKAZE_create()

    features = []

    #frame2, keypoints2, descriptors2も同様に生成
    # keypointsはkeypointオブジェクトのリスト
    # descriptors は、61個の要素が入った1次元配列のリスト
    for frame in frames:
        if pt2 is None:
            pt2 = (frame.shape[1], frame.shape[0])
        
        gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        mask = cv2.rectangle(np.zeros_like(gray), pt1, pt2, color=1, thickness=-1)

        keypoints, descriptors = detector.detectAndCompute(gray, mask=mask)
        features.append([keypoints, descriptors])

    return features  # feature[0]がsrc  feature[1]がtarget

    S

# BFMatcherを用いて特徴（キーポイントとdescription）の突合せを行いマッチするpointを求める
# kp1　　：ターゲット画像のkeypoints
# desc1　：ターゲット画像のdescriptors
# kp2    :ベースとなる画像のkeypoints
# desc2　：ベースとなる画像のdescriptors

# apts = get_match_point(features[0][0], features[0][1], features[1][0], features[1][1])
def get_match_point(kp1, desc1, kp2, desc2):
    
    bf = cv2.BFMatcher()

    # BFMatcherオブジェクト(bf)は
    # bf.Match　　 ：各点に対して最も良いマッチングスコアを持つ対応点のみを返す
    #               従って返ってくる座標は常に1つ
    # bf.knnMatch　：マッチングスコアの上位 k 個の特徴点を返す　※引数のkは何個返すか
                  
    matches = bf.knnMatch(desc1, desc2, k=2)

    # マッチングスコアの上位 k 個の特徴点
    goods = []

    for match1, match2 in matches:
        if match1.distance < 0.7 * match2.distance:
            goods.append(match1)

    target_position = []
    base_position = []

    # 座標取得
    for good in goods:
        target_position.append([kp1[good.queryIdx].pt[0], kp1[good.queryIdx].pt[1]])
        base_position.append([kp2[good.trainIdx].pt[0], kp2[good.trainIdx].pt[1]])

    
    apt1 = np.array(target_position)
    apt2 = np.array(base_position)
    return apt1, apt2



def resize_img(frame, size=(900, 506)):
    return cv2.resize(frame, dsize=size)



if __name__ == "__main__":

    cap = cv2.VideoCapture(movie_path)

    
    ret, frame1 = cap.read()

    H, W =frame1.shape[:2]

    frame1 = resize_img(frame1)

    while cap.isOpened():
        if not ret:
            cap.set(0, 0)
            continue
        
        ret, frame2 = cap.read()
        frame2 = resize_img(frame2)

        frames = [frame1, frame2]

        features = get_features(frames)
        # 引数に注意　ターゲットが第1引数
        adapted_points = get_match_point(features[1][0], features[1][1], features[0][0], features[0][1])

        # M, mask = cv2.findHomography(base_points, target_points, cv2.RANSAC, 5.0)
        M, mask = cv2.findHomography(adapted_points [1], adapted_points [0], cv2.RANSAC, 5.0)
        # frame2_trans = cv2.warpPerspective(frame2, np.linalg.inv(M), (1920, 1080))
        frame2_trans = cv2.warpPerspective(frame2, np.linalg.inv(M), (900, 506))

        frame1 = frame2_trans

        cv2.imshow("frame", frame2_trans)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break


    cap.release()
    cv2.destroyAllWindows()













