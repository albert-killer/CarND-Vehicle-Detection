import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob
import pickle
import os
from moviepy.editor import VideoFileClip


# Function for plotting results
def plot_results(img_before, img_after, name_img_before, name_img_after):

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()

    ax1.imshow(img_before, cmap='gray')
    ax1.set_title(name_img_before, fontsize=40)

    ax2.imshow(img_after, cmap='gray')
    ax2.set_title(name_img_after, fontsize=40)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show()

    return


# Camera calibration and distortion correction
def calibrate_camera(nx, ny, cal_img_path, plot=True):

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((ny*nx, 3), np.float32)
    objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d points in real world space
    imgpoints = []  # 2d points in image plane.

    # Make a list of calibration images
    images = glob.glob(cal_img_path + "calibration*.jpg")

    # Step through the list and search for chessboard corners
    for idx, fname in enumerate(images):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)

            # Draw and display the corners
            cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
            #write_name = 'corners_found'+str(idx)+'.jpg'
            #cv2.imwrite(write_name, img)
            if plot:
                cv2.imshow('img', img)
                cv2.waitKey(500)

    cv2.destroyAllWindows()

    # Test undistortion on an image
    img = cv2.imread(cal_img_path + "calibration1.jpg")
    img_size = (img.shape[1], img.shape[0])

    # Do camera calibration given object points and image points
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)

    dst = cv2.undistort(img, mtx, dist, None, mtx)
    cv2.imwrite(cal_img_path + "test_undist.jpg", dst)

    # Save the camera calibration result for later use (we won't worry about rvecs / tvecs)
    dist_pickle = {}
    dist_pickle["mtx"] = mtx
    dist_pickle["dist"] = dist
    pickle.dump(dist_pickle, open(cal_img_path + "wide_dist_pickle.p", "wb"))
    #dst = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)

    # Visualize undistortion
    if plot:
        plot_results(img, dst, "Original Image", "Undistorted Image")

    return mtx, dist


# Pipeline combining color and gradient thresholding
def thresholding_pipeline(img, adv_grad=False, s_thresh=(90, 255), sxy_thresh=(20, 100),
             mag_thresh=(30, 100), dir_thresh=(0.7, 1.3)):

    img = np.copy(img)
    # Convert to HSV color space and separate the V channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    h_channel = hls[:, :, 0]
    l_channel = hls[:, :, 1]
    s_channel = hls[:, :, 2]

    # Calculate directional gradient
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0)  # Take the derivative in x
    abs_sobelx = np.absolute(sobelx)  # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobelx = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))
    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobelx)
    sxbinary[(scaled_sobelx >= sxy_thresh[0]) & (scaled_sobelx <= sxy_thresh[1])] = 1

    if adv_grad:
        sobely = cv2.Sobel(l_channel, cv2.CV_64F, 0, 1)  # Take the derivative in y
        abs_sobely = np.absolute(sobely)  # Absolute y derivative to accentuate lines away from horizontal
        scaled_sobely = np.uint8(255 * abs_sobely / np.max(abs_sobely))
        # Threshold y gradient
        sybinary = np.zeros_like(scaled_sobely)
        sybinary[(scaled_sobely >= sxy_thresh[0]) & (scaled_sobely <= sxy_thresh[1])] = 1
        # Calculate gradient magnitude
        gradmag = np.sqrt(sobelx ** 2 + sobely ** 2)
        scale_factor = np.max(gradmag) / 255
        gradmag = (gradmag / scale_factor).astype(np.uint8)
        mag_binary = np.zeros_like(gradmag)
        mag_binary[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1
        # Calculate gradient direction
        dir_grad = np.arctan2(abs_sobely, abs_sobelx)
        dir_binary = np.zeros_like(dir_grad)
        dir_binary[(dir_grad >= dir_thresh[0]) & (dir_grad <= dir_thresh[1])] = 1
        # Combine gradient thresholds
        grad_thresh = np.zeros_like(dir_binary)
        grad_thresh[((sxbinary == 1) & (sybinary == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
    else:
        grad_thresh = sxbinary

    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    # TODO: experiment with H channel
    # h_binary = np.zeros_like(h_channel)
    # h_binary[(h_channel >= s_thresh[0]) & (h_channel <= s_thresh[1])] = 1

    # Stack each channel
    # Note color_binary[:, :, 0] is all 0s, effectively an all black image. It might
    # be beneficial to replace this channel with something else.
    color_binary = np.dstack((np.zeros_like(grad_thresh), grad_thresh, s_binary))

    # Combine the two binary thresholds
    combined_binary = np.zeros_like(grad_thresh)
    combined_binary[(s_binary == 1) | (grad_thresh == 1)] = 1

    return combined_binary, color_binary


# Region of interest
def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)

    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    #filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


# Apply perspective transformation to bird's eye view
def perspective_transform(img, src_mask, dst_mask, plot=True):

    img_size = (img.shape[1], img.shape[0])
    src = np.float32(src_mask)
    dst = np.float32(dst_mask)
    M = cv2.getPerspectiveTransform(src, dst)
    warped_img = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)

    # Visualize perspective transform
    if plot:
        src_mask_reshaped = src_mask.reshape((-1, 1, 2))
        dst_mask_reshaped = dst_mask.reshape((-1, 1, 2))
        img_poly = cv2.polylines(img, [src_mask_reshaped], True, (0, 255, 255), thickness=3)
        img_warped_poly = cv2.polylines(warped_img, [dst_mask_reshaped], True, (0, 255, 255), thickness=3)
        plot_results(img_poly, img_warped_poly, "Original Image", "Warped Image")

    return warped_img


# Implement Sliding Windows and Fit a Polynomial
def sliding_windows(binary_warped, nwindows=9, plot=False):
    # TODO: make more robust?
    # Assuming you have created a warped binary image called "binary_warped"
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]/2:, :], axis=0)
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Set height of windows
    window_height = np.int(binary_warped.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2)
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2)
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Visualization
    if plot:
        # Generate x and y values for plotting
        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
        plt.imshow(out_img)
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')
        plt.xlim(0, 1280)
        plt.ylim(720, 0)

        plt.show()

    return left_fit, right_fit


# Skip the sliding windows step once you know where the lines are
def fit_polynomial(binary_warped, left_fit, right_fit, plot=False):
    # Assume you now have a new warped binary image
    # from the next frame of video (also called "binary_warped")
    # It's now much easier to find line pixels!
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 100
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin)))
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    # Visualization
    if plot:
        # Create an image to draw on and an image to show the selection window
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
        window_img = np.zeros_like(out_img)
        # Color in left and right line pixels
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

        # Generate a polygon to illustrate the search window area
        # And recast the x and y points into usable format for cv2.fillPoly()
        left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
        cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
        result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
        plt.imshow(result)
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')
        plt.xlim(0, 1280)
        plt.ylim(720, 0)

        plt.show()

    return left_fit, right_fit, lefty, leftx, righty, rightx


# Measure curvature and return curve radius in meter
def measure_curvature(binary_warped, left_fit, right_fit, lefty, leftx, righty, rightx, bottom_right, bottom_left):
    # Now we have polynomial fits and we can calculate the radius of curvature as follows
    # Define y-value where we want radius of curvature
    # Choose the maximum y-value, corresponding to the bottom of the image
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
    y_eval = np.max(ploty)
    left_curverad = ((1 + (2*left_fit[0]*y_eval + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
    right_curverad = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])
    # print(left_curverad, right_curverad)

    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720  # meters per pixel in y dimension
    xm_per_pix = 3.7/(bottom_right[0] - bottom_left[0])  # meters per pixel in x dimension (160 to 1190 = 1030 px)

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(lefty * ym_per_pix, leftx * xm_per_pix, 2)
    right_fit_cr = np.polyfit(righty * ym_per_pix, rightx * xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    # Now our radius of curvature is in meters
    # print(left_curverad, 'm', right_curverad, 'm')

    return left_curverad, right_curverad


# Warp lane line projection back to original image
def project_lanelines(binary_warped, orig_img, left_fit, right_fit, dst_mask, src_mask, plot=True):

    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 255))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    warped_inv = perspective_transform(color_warp, dst_mask, src_mask, plot=False)

    # Combine the result with the original image
    result = cv2.addWeighted(orig_img, 1, warped_inv, 0.3, 0)

    if plot:
        plot_results(orig_img, result, "Original Image", "Lane Projection")

    return result


# Main lane finding pipeline applying all functions above
def lane_finding_pipeline(input):

    # Step 1 Camera calibration: prepare for distortion correction of images
    calibrate_camera_first_time = False

    if calibrate_camera_first_time:
        mtx, dist = calibrate_camera(9, 6, "camera_cal/", plot=True)
    else:
        dist_pickle = pickle.load(open("camera_cal/wide_dist_pickle.p", "rb"))
        mtx = dist_pickle["mtx"]
        dist = dist_pickle["dist"]

    # Step 2 Distortion correction: Import original image and undistort.
    image = cv2.undistort(input, mtx, dist, None, mtx)
    # plot_results(input, image, "Original Image", "Undistorted Image")

    # Step 3 Thresholding: various combinations of color and gradient thresholds to generate a binary image
    # TODO: experiment with thresholds, when using different input images
    binary_image, color_binary = thresholding_pipeline(image, adv_grad=False, s_thresh=(90, 255), sxy_thresh=(20, 100))
    # plot_results(color_binary, binary_image, "Color+gradient thresh", "Binary Image")

    # Step 4.1 Perspective transform on original image:
    # TODO: following values are optimized on test image 'straight_lines2.jpg'
    top_left = [540, 480]
    top_right = [754, 480]
    bottom_right = [1190, 718]
    bottom_left = [160, 718]
    src_mask = np.array([[(top_left[0], top_left[1]),
                          (top_right[0], top_right[1]),
                          (bottom_right[0], bottom_right[1]),
                          (bottom_left[0], bottom_left[1])]], np.int32)
    dst_mask = np.array([[(bottom_left[0], 0),
                          (bottom_right[0], 0),
                          (bottom_right[0], bottom_right[1]),
                          (bottom_left[0], bottom_left[1])]], np.int32)
    image_warped = perspective_transform(image, src_mask, dst_mask, plot=False)

    # Step 4.2 Perspective transform on binary image:
    binary_warped = perspective_transform(binary_image, src_mask, dst_mask, plot=False)
    #binary_image = region_of_interest(binary_image, reg_mask)

    # Step 5 Fit Polynomial
    left_fit, right_fit = sliding_windows(binary_warped, nwindows=9, plot=False)
    # TODO: Save polynomials from first frame in order to skip sliding windows for following frames
    left_fit, right_fit, lefty, leftx, righty, rightx = fit_polynomial(binary_warped, left_fit, right_fit, plot=False)

    # Step 6 Measure curve radius and distance from center
    left_curverad, right_curverad = measure_curvature(binary_warped, left_fit, right_fit, lefty, leftx, righty, rightx,
                                                      bottom_right, bottom_left)
    # Get car distance from center, assuming camera is mounted at the center of the car
    lane_center = leftx[0] + ((rightx[0] - leftx[0]) / 2)
    camera_center = input.shape[1] / 2
    distance_center = (camera_center - lane_center) * (3.7 / (bottom_right[0] - bottom_left[0]))
    # Draw line for lane center
    cv2.line(image, (int(lane_center), 720), (int(lane_center), 620), (255, 0, 0), 2)
    # Draw line for camera center
    cv2.line(image, (int(camera_center), 720), (int(camera_center), 670), (0, 0, 255), 2)

    # Text output on image
    if left_curverad < 10000.0:
        text_output_1 = str(round(left_curverad, 2))
    else:
        text_output_1 = "straight"
    cv2.putText(image, text_output_1, (65, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), thickness=2)
    text_output_2 = "| radius of curvature [m]"
    cv2.putText(image, text_output_2, (275, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 185, 34), thickness=2)
    text_output_3 = str(round(abs(distance_center), 2))
    cv2.putText(image, text_output_3, (155, 130), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (242, 98, 84), thickness=2)
    side = ""
    if distance_center > 0:
        side = ", to right"
    elif distance_center < 0:
        side = ", to left"
    text_output_4 = "| distance from center [m]" + side
    cv2.putText(image, text_output_4, (275, 130), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 185, 34), thickness=2)

    # Step 7 Project Lines
    result = project_lanelines(binary_warped, image, left_fit, right_fit, dst_mask, src_mask, plot=False)

    return result