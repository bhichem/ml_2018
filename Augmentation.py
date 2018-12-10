import cv2
import numpy as np
import os
import skimage.transform
from sklearn.model_selection import train_test_split

def main():
    getImageFileNames()


def getImageFileNames():
    PATH = os.getcwd()
    data_path = PATH + '/Dataset'
    data_dir_list = os.listdir(data_path)

    img_data_list = []
    labels = []
    folderCounter = 0

    for eachfolderName in data_dir_list:
        #print(eachfolderName)
        img_list = os.listdir(data_path + '/' + eachfolderName)

        img_data_list.append(img_list)
        labels.append(eachfolderName)
        imageCounter = 0

        # for eachImage in img_list:
        #     print(img_data_list[folderCounter][imageCounter])
        #     retrieve_dir = data_path + "\\" + labels[folderCounter] + "\\" + img_data_list[folderCounter][imageCounter]
        #     image = cv2.imread(retrieve_dir)
        #     flip_images(image, data_path, labels, folderCounter, img_data_list, imageCounter)
        #     rotate(image, data_path, labels, folderCounter, img_data_list, imageCounter, 10, 'rotate_minus_10')
        #     rotate(image, data_path, labels, folderCounter, img_data_list, imageCounter, -10, 'rotate_plus_10')
        #     gaussian_noise(image, data_path, labels, folderCounter, img_data_list, imageCounter)
        #     lighting(image, data_path, labels, folderCounter, img_data_list, imageCounter, 0.5, 'Darker')
        #     lighting(image, data_path, labels, folderCounter, img_data_list, imageCounter, 1.5, 'Lighter')
        #     imageCounter = imageCounter + 1;
        folderCounter = folderCounter + 1;
    XTrain, XTest, YTrain, YTest = train_test_split(img_data_list, labels, test_size=0.2, random_state=42, shuffle=True)
    print(len(XTrain))
    print(len(XTest))


def flip_images(image, data_path, labels, folderCounter, img_data_list, imageCounter):
    save_dir_flipped = data_path + "\\" + labels[folderCounter] + "\\" + 'flipped' + img_data_list[folderCounter][
        imageCounter]
    flipped_image = np.fliplr(image)
    cv2.imwrite(os.path.join(save_dir_flipped), flipped_image)


def rotate(image, data_path, labels, folderCounter, img_data_list, imageCounter, angle, dir_name):
    save_dir_rotate = data_path + "\\" + labels[folderCounter] + "\\" + dir_name + img_data_list[folderCounter][
        imageCounter]
    rotated_image = skimage.transform.rotate(image, angle= angle, preserve_range=True).astype(np.uint8)
    cv2.imwrite(os.path.join(save_dir_rotate), rotated_image)


def gaussian_noise(image, data_path, labels, folderCounter, img_data_list, imageCounter):
    save_dir_flipped = data_path + "\\" + labels[folderCounter] + "\\" + 'gaussian_noise_' + img_data_list[folderCounter][
        imageCounter]
    mean = 20;
    std = 20;
    noisy_img = image + np.random.normal(mean, std, image.shape)
    noisy_img_clipped = np.clip(noisy_img, 0, 255)
    cv2.imwrite(os.path.join(save_dir_flipped), noisy_img_clipped)


def lighting(image, data_path, labels, folderCounter, img_data_list, imageCounter, gamma, darkOrLight):
   save_dir_lighting= data_path + "\\" + labels[folderCounter] + "\\" + darkOrLight + img_data_list[folderCounter][imageCounter]
   invGamma = 1.0 / gamma
   table = np.array([((i / 255.0) ** invGamma) * 255
      for i in np.arange(0, 256)]).astype("uint8")
   flippedLighting = cv2.LUT(image, table)
   cv2.imwrite(os.path.join(save_dir_lighting), flippedLighting)


if __name__ == '__main__':
    main()
