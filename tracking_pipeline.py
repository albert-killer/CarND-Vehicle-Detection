import glob
import time
import os
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.cross_validation import train_test_split  # for sklearn > 0.17 use sklearn.model_selection instead
from moviepy.editor import VideoFileClip
from my_lesson_functions import *
from lane_finding import *


# Define a function to extract features from a single image window
# This function is very similar to extract_features()
# just for a single image rather than list of images
def single_img_features(img, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9,
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):

    # 1) Define an empty list to receive features
    img_features = []
    # 2) Apply color conversion if other than 'RGB'
    if color_space != 'RGB':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    else: feature_image = np.copy(img)
    # 3) Compute spatial features if flag is set
    if spatial_feat == True:
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        # 4) Append features to list
        img_features.append(spatial_features)
    # 5) Compute histogram features if flag is set
    if hist_feat == True:
        hist_features = color_hist(feature_image, nbins=hist_bins)
        # 6) Append features to list
        img_features.append(hist_features)
    # 7) Compute HOG features if flag is set
    if hog_feat == True:
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.extend(get_hog_features(feature_image[:,:,channel],
                                    orient, pix_per_cell, cell_per_block,
                                    vis=False, feature_vec=True))
        else:
            hog_features = get_hog_features(feature_image[:, :, hog_channel], orient,
                                            pix_per_cell, cell_per_block, vis=False, feature_vec=True)
        # 8) Append features to list
        img_features.append(hog_features)

    #9) Return concatenated array of features
    return np.concatenate(img_features)


# Define a function you will pass an image
# and the list of windows to be searched (output of slide_windows())
def search_windows(img, windows, clf, scaler, color_space='RGB',
                    spatial_size=(32, 32), hist_bins=32,
                    hist_range=(0, 256), orient=9,
                    pix_per_cell=8, cell_per_block=2,
                    hog_channel=0, spatial_feat=True,
                    hist_feat=True, hog_feat=True):

    # 1) Create an empty list to receive positive detection windows
    on_windows = []
    # 2) Iterate over all windows in the list
    for window in windows:
        # 3) Extract the test window from original image
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))
        # 4) Extract features for that window using single_img_features()
        features = single_img_features(test_img, color_space=color_space,
                            spatial_size=spatial_size, hist_bins=hist_bins,
                            orient=orient, pix_per_cell=pix_per_cell,
                            cell_per_block=cell_per_block,
                            hog_channel=hog_channel, spatial_feat=spatial_feat,
                            hist_feat=hist_feat, hog_feat=hog_feat)
        # 5) Scale extracted features to be fed to classifier
        test_features = scaler.transform(np.array(features).reshape(1, -1))
        # 6) Predict using your classifier
        prediction = clf.predict(test_features)
        # 7) If positive (prediction == 1) then save the window
        if prediction == 1:
            on_windows.append(window)
    # 8) Return windows for positive detections
    return on_windows


# Image processing for video input
def process_video(img):

    # NOTE: The output you return should be a color image (3 channel) for processing video below
    file_dir = "frames.jpg"
    mpimg.imsave(file_dir, img)
    image = mpimg.imread(file_dir)
    draw_image = np.copy(image)

    # Create a list of sliding windows
    windows = slide_window(image, x_start_stop=x_start_stop, y_start_stop=y_start_stop,
                           xy_window=xy_window, xy_overlap=xy_overlap)

    # Search for vehicles and get a list of positive detections
    hot_windows = search_windows(image, windows, svc, X_scaler, color_space=color_space,
                                 spatial_size=spatial_size, hist_bins=hist_bins,
                                 orient=orient, pix_per_cell=pix_per_cell,
                                 cell_per_block=cell_per_block,
                                 hog_channel=hog_channel, spatial_feat=spatial_feat,
                                 hist_feat=hist_feat, hog_feat=hog_feat)

    # Draw all positive detections
    window_img = draw_boxes(draw_image, hot_windows, line_color=line_color_2, line_thickness=1)

    # Start applying heat map to reduce multi detections and false positives
    heatmap = np.zeros_like(image[:, :, 0]).astype(np.float)
    # Add heat to each box in box list
    heatmap = add_heat(heatmap, hot_windows)

    # Apply threshold to help remove false positives
    heatmap = apply_threshold(heatmap, heat_thresh)

    # Apply Multi-frame accumulated heat map
    # Inspired by my mentor and different threads from the Udacity's Atlassian forum
    heatmap_buffer.append(heatmap)
    # Limit the heat map buffer to a certain amount...
    if len(heatmap_buffer) > 8:
        # ... by cutting off the oldest heatmap after buffer exceeds its limit
        heatmap_buffer.pop(0)
    # Then start with the next oldest heatmap within the buffer
    true_positives = heatmap_buffer[0]
    # Iterate through the heat map buffer
    for current_heatmap in heatmap_buffer:
        # Comparing each subsequent heat map to the one representing previous detections
        # Favoring overlaps by using logical AND comparison
        true_positives = np.logical_and(true_positives, current_heatmap)
    # As a result new detections get classified as "true positives" sooner the lower the limit above is set
    # Low limit leads to more false positives but early response in detection of new vehicles
    # High limits reduce false positives but delay detection of new vehicles
    heatmap = true_positives

    # Apply threshold to help remove false positives
    heatmap = apply_threshold(heatmap, heat_thresh)

    # Find final bounding boxes from heatmap using label function
    labels = label(heatmap)

    # Draw image with lane area from Project "Advanced Lane Finding"
    lane_lines_img = lane_finding_pipeline(window_img)
    # Fill bounding boxes of detected vehicles
    filled_bboxes_img = draw_labeled_bboxes(np.copy(lane_lines_img), labels, line_color, line_thickness=-3)
    # Add bounding boxes of detected vehicles after heatmap process
    image_export = draw_labeled_bboxes(np.copy(lane_lines_img), labels, line_color, line_thickness)

    return cv2.addWeighted(image_export, 1, filled_bboxes_img, 0.3, -30)


# IMPORT SMALL IMAGE DATA SET FOR TRAINING CLASSIFIER
cars = []
notcars = []
# Read in vehicle samples
images_car = glob.glob("dataset/vehicles_smallset/**/*.jpeg")
for image in images_car:
    cars.append(image)
# Read in non-vehicle samples
images_notcar = glob.glob("dataset/non-vehicles_smallset/**/*.jpeg")
for image in images_notcar:
    notcars.append(image)


'''''''''
# IMPORT LARGE IMAGE DATA SET FOR TRAINING CLASSIFIER
cars = []
notcars = []
# Read in cars
images_car = glob.glob("dataset/vehicles/**/*.png")
for image in images_car:
    cars.append(image)
# Read in notcars
images_notcar = glob.glob("dataset/non-vehicles/**/*.png")
for image in images_notcar:
    notcars.append(image)

# Reduce the sample size for faster testing:
# sample_size = 2000
# cars = cars[0:sample_size]
# notcars = notcars[0:sample_size]
'''''''''


# SET PARAMETERS FOR FEATURE EXTRACTION
### TODO: Tweak these parameters for optimization and visualization.
color_space = 'YCrCb'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9  # HOG orientations, usually between 6 and 12
pix_per_cell = 14  # HOG pixels per cell
cell_per_block = 2  # HOG cells per block
hog_channel = 1  # Can be 0, 1, 2, or "ALL"
spatial_size = (32, 32)  # Spatial binning dimensions
hist_bins = 196  # Number of histogram bins
spatial_feat = True  # Spatial features on or off
hist_feat = True  # Histogram features on or off
hog_feat = True  # HOG features on or off
y_start_stop = [400, 700]  # Min and max in y to search in slide_window()
x_start_stop = [380, 1280]  # Min and max in x to search in slide_window()
xy_window = (80, 80)  # size of sliding windows
xy_overlap = (0.55, 0.55)  # overlap fraction of sliding windows
line_color = (0, 255, 102)  # i.e. turquoise : (0, 255, 102), pink: (253, 43, 255)
line_color_2 = (147, 164, 158)  # i.e. light gray: (147, 164, 158), light cyan: (127, 222, 187)
line_thickness = 3  # lines of bounding boxes marking detected vehicles
heat_thresh = 0.75  # threshold for heatmap filter process

# Global variables for advanced heatmap processing
heatmap_buffer = []
global heatmap_sum
heatmap_sum = np.zeros((720, 1280)).astype(np.float64)

# Extract features using above defined parameters
car_features = extract_features(cars, color_space=color_space,
                                spatial_size=spatial_size, hist_bins=hist_bins,
                                orient=orient, pix_per_cell=pix_per_cell,
                                cell_per_block=cell_per_block,
                                hog_channel=hog_channel, spatial_feat=spatial_feat,
                                hist_feat=hist_feat, hog_feat=hog_feat)
notcar_features = extract_features(notcars, color_space=color_space,
                                   spatial_size=spatial_size, hist_bins=hist_bins,
                                   orient=orient, pix_per_cell=pix_per_cell,
                                   cell_per_block=cell_per_block,
                                   hog_channel=hog_channel, spatial_feat=spatial_feat,
                                   hist_feat=hist_feat, hog_feat=hog_feat)


# CREATE DATA SETS
X = np.vstack((car_features, notcar_features)).astype(np.float64)
# Fit a per-column scaler
X_scaler = StandardScaler().fit(X)
# Apply the scaler to X
scaled_X = X_scaler.transform(X)

# Define the labels vector
y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

# Split up data into randomized training and test sets
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(
    scaled_X, y, test_size=0.2, random_state=rand_state)

print('Using:', orient, 'orientations', pix_per_cell,
      'pixels per cell and', cell_per_block, 'cells per block')
print('Feature vector length:', len(X_train[0]))
print('X_train length total:', len(X_train))

# CREATE AND TRAIN CLASSIFIER
# Use a linear SVC
svc = LinearSVC()
# Check the training time for the SVC
t = time.time()
svc.fit(X_train, y_train)
t2 = time.time()
print(round(t2-t, 2), 'Seconds to train SVC...')
# Check the score of the SVC
print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
# Check the prediction time for a single sample
t = time.time()


# APPLY TRACKING PIPELINE
video_input = True  # Switch between video and image input mode

if video_input:
    clip_input = VideoFileClip("project_video.mp4")
    clip_output = clip_input.fl_image(process_video)  # NOTE: this function expects color images!!
    clip_output.write_videofile("project_video_detection.mp4", audio=False)
else:
    file_list = []
    test_images = glob.glob("test_images/*.jpg")
    for img in test_images:
        file_list.append(img)
    # file_list = os.listdir("test_images/*.jpg")  # import all image of test folder
    print("Processing images from folder test_images.")
    for file_name in file_list:
        image = mpimg.imread(file_name)
        draw_image = np.copy(image)
        windows = slide_window(image, x_start_stop=x_start_stop, y_start_stop=y_start_stop,
                               xy_window=xy_window, xy_overlap=xy_overlap)

        hot_windows = search_windows(image, windows, svc, X_scaler, color_space=color_space,
                                 spatial_size=spatial_size, hist_bins=hist_bins,
                                 orient=orient, pix_per_cell=pix_per_cell,
                                 cell_per_block=cell_per_block,
                                 hog_channel=hog_channel, spatial_feat=spatial_feat,
                                 hist_feat=hist_feat, hog_feat=hog_feat)

        window_img = draw_boxes(draw_image, hot_windows, line_color=line_color, line_thickness=line_thickness)
        # export new image
        file_name_only = file_name.split('/')[1]
        file_dir = 'output_images/' + file_name_only
        mpimg.imsave(file_dir, window_img)
        print(file_name_only, "exported.")

        # improve via heatmap
        if 0:
            improved_image, heatmap = remove_false_positives(image, hot_windows, heat_thresh=heat_thresh,
                                                             line_color=line_color)
            # export new image
            file_name_only = file_name_only.split('.')[0] + '_improved.jpg'
            file_dir = 'output_images/' + file_name_only
            mpimg.imsave(file_dir, improved_image)
            print(file_name_only, "exported.")
    print("All test images processed and exported.")

    # Further plotting for producing output images for project report
    # TODO: delete this section after report is submitted
    if 0:
        file_list = []
        test_images = glob.glob("output_images/*.jpg")
        for img in test_images:
            file_list.append(img)

        fig = plt.figure()

        plt.subplot(231)
        plt.imshow(mpimg.imread(file_list[3]), cmap='gray')
        plt.title(' ')

        plt.subplot(232)
        plt.imshow(mpimg.imread(file_list[5]), cmap='gray')
        plt.title(' ')

        plt.subplot(233)
        plt.imshow(mpimg.imread(file_list[1]), cmap='gray')
        plt.title(' ')

        plt.subplot(234)
        plt.imshow(mpimg.imread(file_list[0]), cmap='gray')
        plt.title(' ')

        plt.subplot(235)
        plt.imshow(mpimg.imread(file_list[4]), cmap='gray')
        plt.title(' ')

        plt.subplot(236)
        plt.imshow(mpimg.imread(file_list[2]), cmap='gray')
        plt.title(' ')

        plt.show()
