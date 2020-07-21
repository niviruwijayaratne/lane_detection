import matplotlib.pyplot as plt
import cv2
import numpy as np
import glob
from moviepy.editor import VideoFileClip
from vid_manipulation import create_frames, create_vid 
from detect import detect
# from IPython.display import HTML


files = glob.glob('camera_cal/calibration*.jpg')
image_points = []
mapped_points = []
grid = np.zeros((54, 3), np.float32)
grid[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1,2)

for fname in files:
    image = cv2.imread(fname)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    ret_val, corners = cv2.findChessboardCorners(gray, (9, 6), None)

    if ret_val:
        image_points.append(corners)
        mapped_points.append(grid)
        drawn = cv2.drawChessboardCorners(image, (9, 6), corners, ret_val)
        cv2.imshow('drawn', drawn)
        cv2.waitKey(2)
ret, cam_mat, dist_coeffs, rot_vector, trans_vector = cv2.calibrateCamera(mapped_points, image_points, (image.shape[1], image.shape[0]),None,None)

def undistort(image):
    undistorted = cv2.undistort(image, cam_mat, dist_coeffs, None, cam_mat)
    return undistorted

def threshold(image, sobel_angle_thresh=[0.4, 1.5], sobel_x_thresh = 20, h_thresh=[None, 30], s_thresh=[110, 255]):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=15)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=15)

    xmag = np.sqrt(np.square(sobelx))
    x_eight_bit = np.uint8(xmag*255/np.max(xmag))
    
    gradient_angles = np.arctan2(np.absolute(sobely), np.absolute(sobelx))

    zeros = np.zeros_like(gradient_angles)
    zeros[(gradient_angles <= sobel_angle_thresh[1]) & (gradient_angles >= sobel_angle_thresh[0])  & (x_eight_bit > sobel_x_thresh)] = 1
    
    hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    h = hls[:, :, 0]
    # l = hls[:, :, 1]
    s = hls[:,:, 2]
    
    zeros[(s > s_thresh[0]) & (s < s_thresh[1]) & (h < h_thresh[1])] = 1
    return zeros


def warp(image, src=np.array([[210, 720], [610, 450], [720, 458] , [1125, 720]], np.float32), dst=np.array([[300, 720], [300, 0], [940, 0],[940, 720]], np.float32), inter_method=cv2.INTER_LINEAR):
    y = image.shape[0]
    x = image.shape[1]
    if src.any()==None:
        src_points = np.array([[210, y], [610, 450], [720, 458] , [1125, y]], np.float32)
    else:
        src_points = src

    if dst.any()==None:
        dst_points = np.array([[300, y], [300, 0], [940, 0],[940, y]], np.float32)
    else:
        dst_points = dst

    persp_transform_mat = cv2.getPerspectiveTransform(src_points, dst_points)
    warped_img = cv2.warpPerspective(image, persp_transform_mat, (x, y), flags=inter_method)
    return warped_img



def first_fit(image, n_windows=9, marg=125, min_pixels_recenter=50):
    bottom_half = image[image.shape[0]//2:, :]
    canvas = np.dstack((image, image, image))*255
    histogram = np.sum(bottom_half, axis=0)

    midpoint = np.int(histogram.shape[0]//2)
    left_start = np.argmax(histogram[250:midpoint]) + 250
    right_start = np.argmax(histogram[midpoint:]) + midpoint

    num_windows = n_windows
    margin = marg
    min_num_pixels = min_pixels_recenter
    window_height = image.shape[0]//num_windows

    nonzero = image.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    
    left_curr = left_start
    right_curr = right_start
    left_indicesx = []
    left_indicesy = []
    right_indicesx = []
    right_indicesy = []
    curr_height = canvas.shape[0]
    
    for window in range(num_windows):
        cv2.rectangle(canvas, (left_curr-margin, curr_height), 
                    (left_curr + margin, curr_height - window_height), (0,255,0), 5)

        left_conditionx = nonzerox[(nonzerox >= left_curr-margin) & (nonzerox < left_curr + margin) & 
                                (nonzeroy >= curr_height-window_height) & (nonzeroy < curr_height)]
        left_conditiony = nonzeroy[(nonzerox >= left_curr-margin) & (nonzerox < left_curr + margin) & 
                                (nonzeroy >= curr_height-window_height) & (nonzeroy < curr_height)]
        len_left = len(left_conditionx)
        left_indicesx.append(left_conditionx)
        left_indicesy.append(left_conditiony)
        if (len_left > min_num_pixels):
            left_curr = np.int(np.mean(left_indicesx[-1]))

        cv2.rectangle(canvas, (right_curr-margin, curr_height), 
                    (right_curr + margin, curr_height - window_height), (0,255,0), 5)
        right_conditionx = nonzerox[(nonzerox >= right_curr-margin) & (nonzerox < right_curr + margin) & 
                                    (nonzeroy >= curr_height-window_height) & (nonzeroy < curr_height)]
        right_conditiony = nonzeroy[(nonzerox >= right_curr-margin) & (nonzerox < right_curr + margin) & 
                                    (nonzeroy >= curr_height-window_height) & (nonzeroy < curr_height)]

        len_right = len(right_conditionx)  
        right_indicesx.append(right_conditionx)
        right_indicesy.append(right_conditiony)
        if (len_right > min_num_pixels):
            right_curr = np.int(np.mean(right_indicesx[-1]))

        curr_height -= window_height

    leftx = np.concatenate(left_indicesx)
    lefty = np.concatenate(left_indicesy)
    rightx = np.concatenate(right_indicesx)
    righty = np.concatenate(right_indicesy)

    print(leftx, lefty, rightx, righty)
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    # plot_vals = np.linspace(0, canvas.shape[0] -1, canvas.shape[0])

    # leftplot = left_fit[0]*plot_vals**2 + left_fit[1]*plot_vals + left_fit[2]
    # rightplot = right_fit[0]*plot_vals**2 + right_fit[1]*plot_vals + right_fit[2]

    return left_fit, right_fit


def search(image, left_fit, right_fit):
    nonzero = image.nonzero()
    nonzerox = nonzero[1]
    nonzeroy = nonzero[0]

    margin = 100
    
    leftx = nonzerox[(nonzerox > ((left_fit[0]*nonzeroy**2) + (left_fit[1]*nonzeroy) + (left_fit[2]) - margin)) & 
                        (nonzerox < ((left_fit[0]*nonzeroy**2) + (left_fit[1]*nonzeroy) + (left_fit[2]) + margin))]
    lefty = nonzeroy[(nonzerox > ((left_fit[0]*nonzeroy**2) + (left_fit[1]*nonzeroy) + (left_fit[2]) - margin)) & 
                        (nonzerox < ((left_fit[0]*nonzeroy**2) + (left_fit[1]*nonzeroy) + (left_fit[2]) + margin))]
    rightx = nonzerox[(nonzerox > ((right_fit[0]*nonzeroy**2) + (right_fit[1]*nonzeroy) + (right_fit[2]) - margin)) & 
                        (nonzerox < ((right_fit[0]*nonzeroy**2) + (right_fit[1]*nonzeroy) + (right_fit[2]) + margin))]
    righty = nonzeroy[(nonzerox > ((right_fit[0]*nonzeroy**2) + (right_fit[1]*nonzeroy) + (right_fit[2]) - margin)) & 
                        (nonzerox < ((right_fit[0]*nonzeroy**2) + (right_fit[1]*nonzeroy) + (right_fit[2]) + margin))]
    
    result = np.dstack((image, image, image))*255
    background = np.zeros_like(result)
    

    ploty = np.linspace(0, image.shape[0] - 1, image.shape[0])

    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    result[lefty, leftx] = (0,0,255)
    result[righty, rightx] = (255,0,0)
    
    lower_left = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
    upper_left = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, 
                            ploty])))])

    left_line = np.hstack((lower_left, upper_left))


    lower_right = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
    upper_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, 
                            ploty])))])
    right_line = np.hstack((lower_right, upper_right))

    
    
    
    # left_lane_lower = 


    average_left = np.array([np.transpose(np.vstack([left_fitx+6, ploty]))])
    average_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx-6, ploty])))])


    average_left2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx + 6, ploty])))])
    average_right2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx + 30, ploty])))])
    left = np.array([np.transpose(np.vstack([left_fitx-20, ploty]))])
    right = np.array([np.transpose(np.vstack([right_fitx-10, 
                            ploty]))])
    leftlanebound = np.hstack((left, average_left2))
    rightlanebound = np.hstack((average_right2, right))

    # cv2.fillPoly(background, np.int_(leftlanebound), (255,0,0))

    bounds = np.hstack((average_left, average_right))
    cv2.fillPoly(background, np.int_([bounds]), (0,255,0))

    # cv2.fillPoly(background, np.int_([leftlanebound]), (255,0, 0))
    # cv2.fillPoly(background, np.int_([rightlanebound]), (0,0, 255))
    result = cv2.addWeighted(result, 1, background, 0.3, 0)
    
    # plt.plot(left_fitx, ploty, color='yellow')
    # plt.plot(right_fitx, ploty, color='yellow')
    
    return result, left_fitx, right_fitx, ploty, average_left, average_right, left_fit, right_fit, leftlanebound, rightlanebound

def calc_curverad(image, left_fit, right_fit):
    ploty = np.linspace(0, image.shape[0]-1, image.shape[0])
    y_conv = (30/720)
    x_conv = (3.7/700)
    left_fit *= x_conv
    right_fit *= x_conv
    ploty*= y_conv
    
    leftfit = np.polyfit(ploty, left_fit, 2)
    rightfit = np.polyfit(ploty, right_fit, 2)
    
    max_y = np.max(ploty)
    
    rcurve_left = ((1 + np.square(2*leftfit[0]*max_y) + leftfit[1])**(3/2))/np.abs(leftfit[0])
    rcurve_right = ((1 + np.square(2*rightfit[0]*max_y) + rightfit[1])**(3/2))/np.abs(rightfit[0])
    
    return rcurve_left, rcurve_right



def pipeline(vid_path, image_path):
    frame_curr, image_path = create_frames(vid_path, image_path)
    fnames = glob.glob(image_path + '/*.jpg')
    index = 0
    left_fit = None
    right_fit = None
    for i in range(len(fnames)):
        filename = image_path + '/' + str(i) + '.jpg'
        print('doing operations on' + filename)
        # tested_image = cv2.imread(filename)
        tested_image = detect(filename)
        undistorted_image = undistort(tested_image)
        thresholded_image = threshold(undistorted_image)

        thresh_small = np.dstack((thresholded_image,thresholded_image,thresholded_image))*255
        # s = np.array([[317, 527], [593, 335], [745, 334] , [1094, 531]], np.float32)
        # d = np.array([[400, 720], [400, 0], [1000, 0],[1000, 720]], np.float32)
        src_points = np.array([[285, 560], [595, 334], [733, 327], [1111, 540]], np.float32)
        dst_points = np.array([[300, 720], [300, 0], [940, 0],[940, 720]], np.float32)
        warped_image = warp(thresholded_image, src=src_points, dst=dst_points)
        warped_small = np.dstack((warped_image,warped_image,warped_image))*255
        if index == 0:
            lane_fits = first_fit(warped_image)
            left_fit, right_fit = lane_fits[0], lane_fits[1]
        index += 1
        searched = search(warped_image, left_fit, right_fit)
        left_fit, right_fit = searched[6], searched[7]
        rad = calc_curverad(searched[0], searched[1], searched[2])
        warp_zero = np.zeros_like(warped_image).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
        color_warp2 = np.dstack((warp_zero, warp_zero, warp_zero))
        left_bound = searched[4]
        right_bound = searched[5]
        bounds = np.hstack((left_bound, right_bound))

        cv2.fillPoly(color_warp2, np.int_(searched[8]), (0,0,255))
        cv2.fillPoly(color_warp2, np.int_(searched[9]), (255,0,0))
        cv2.fillPoly(color_warp, np.int_([bounds]), (0,255, 0))
        # src_points = np.array([[285, 560], [595, 334], [733, 327], [1111, 540]], np.float32)
        # dst_points = np.array([[300, 720], [300, 0], [940, 0],[940, 720]], np.float32)
        
        
        persp_transform_mat_inv = cv2.getPerspectiveTransform(dst_points, src_points)
        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        background = cv2.warpPerspective(color_warp, persp_transform_mat_inv, (image.shape[1], image.shape[0])) 
        background2 = cv2.warpPerspective(color_warp2, persp_transform_mat_inv, (image.shape[1], image.shape[0])) 
        # Combine the result with the original image
        result = cv2.addWeighted(undistorted_image, 1, background, 0.3, 0)
        result2 = cv2.addWeighted(result, 1, background2, 1, 0)

        s_img = cv2.resize(thresh_small, (0,0), fx=0.2, fy=0.2) 
        x_offset=y_offset=10
        result2[y_offset:y_offset+s_img.shape[0], x_offset:x_offset+s_img.shape[1]] = s_img

        x_offset2, y_offset2=50 + s_img.shape[1], 10
        s_img2 = cv2.resize(warped_small, (0,0), fx=0.2, fy=0.2)
        result2[y_offset2:y_offset2+s_img2.shape[0], x_offset2:x_offset2+s_img2.shape[1]] = s_img2

        x_offset3, y_offset3 = 90 + s_img.shape[1] + s_img2.shape[1], 10
        s_img3 = cv2.resize(searched[0], (0,0), fx=0.2, fy=0.2)
        result2[y_offset3:y_offset3+s_img3.shape[0], x_offset3:x_offset3+s_img3.shape[1]] = s_img3

        text = "Lane Curvatures:"
        text2 = "Left Lane = " + str("{:.6f}".format(rad[0]))
        text3 = "Right Lane = " + str("{:.6f}".format(rad[1]))

        result2 = cv2.putText(result2, text, (130 + s_img.shape[1] + s_img2.shape[1] + s_img3.shape[1], 30), cv2.FONT_HERSHEY_SIMPLEX,  
                        0.85, (255, 255, 255) , 2, cv2.LINE_AA) 
        result2 = cv2.putText(result2, text2, (130 + s_img.shape[1] + s_img2.shape[1] + s_img3.shape[1], 90), cv2.FONT_HERSHEY_SIMPLEX,  
                        0.85, (255, 255, 255) , 2, cv2.LINE_AA) 
        result2 = cv2.putText(result2, text3, (130 + s_img.shape[1] + s_img2.shape[1] + s_img3.shape[1], 150), cv2.FONT_HERSHEY_SIMPLEX,  
                        0.85, (255, 255, 255) , 2, cv2.LINE_AA) 

        cv2.imwrite(image_path + '/' + str(i) + '.jpg', result2)
    
    return create_vid(frame_curr, '../final.mp4', image_path)

pipeline('../720real.mp4', '../vid_frames')


# tested_image = cv2.imread('test_images/test2.jpg')
# undistorted_image = undistort(tested_image)
# thresholded_image = threshold(undistorted_image)
# src_pts = np.array([[300, 720], [580, 450], [662, 449] , [1125, 720]], np.float32)
# dst_points = np.array([[300, 720], [300, 0], [900, 0],[900, 720]], np.float32)
# f = np.dstack((thresholded_image, thresholded_image, thresholded_image))*255
# warped_image = warp(thresholded_image, src = src_pts, dst = dst_points)
# p = np.dstack((warped_image, warped_image, warped_image))*255
# # print(len(warped_image[warped_image.nonzero()]))
# # print(len(warped_image))
# # print(warped_image)

# # cv2.line(undistorted_image, (src_pts[0][0], src_pts[0][1]), (src_pts[1][0], src_pts[1][1]), (255,0,0), 5)
# # cv2.line(undistorted_image, (src_pts[1][0], src_pts[1][1]), (src_pts[2][0], src_pts[2][1]), (255,0,0), 5)
# # cv2.line(undistorted_image, (src_pts[2][0], src_pts[2][1]), (src_pts[3][0], src_pts[3][1]), (255,0,0), 5)
# # cv2.line(undistorted_image, (src_pts[0][0], src_pts[0][1]), (src_pts[3][0], src_pts[3][1]), (255,0,0), 5)

# # cv2.line(warped_image, (d[0][0], d[0][1]), (d[1][0], d[1][1]), (255,0,0), 5)
# # cv2.line(warped_image, (d[1][0], d[1][1]), (d[2][0], d[2][1]), (255,0,0), 5)
# # cv2.line(warped_image, (d[2][0], d[2][1]), (d[3][0], d[3][1]), (255,0,0), 5)
# # cv2.line(warped_image, (d[0][0], d[0][1]), (d[3][0], d[3][1]), (255,0,0), 5)

# lane_fits = first_fit(warped_image)
# left_fit, right_fit = lane_fits[0], lane_fits[1]
# # index += 1
# searched = search(warped_image, left_fit, right_fit)
# rad = calc_curverad(searched[0], searched[1], searched[2])
# warp_zero = np.zeros_like(warped_image).astype(np.uint8)
# color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
# color_warp2 = np.dstack((warp_zero, warp_zero, warp_zero))

# left_bound = searched[4]
# right_bound = searched[5]
# bounds = np.hstack((left_bound, right_bound))

# cv2.fillPoly(color_warp2, np.int_(searched[8]), (255,0,0))
# cv2.fillPoly(color_warp2, np.int_(searched[9]), (0,0,255))
# cv2.fillPoly(color_warp, np.int_([bounds]), (0,255, 0))
# src_points = np.array([[210, 720], [610, 450], [720, 458] , [1125, 720]], np.float32)
# dst_points = np.array([[300, 720], [300, 0], [940, 0],[940, 720]], np.float32)

# persp_transform_mat_inv = cv2.getPerspectiveTransform(dst_points, src_points)
# # Warp the blank back to original image space using inverse perspective matrix (Minv)
# background = cv2.warpPerspective(color_warp, persp_transform_mat_inv, (image.shape[1], image.shape[0])) 
# background2 = cv2.warpPerspective(color_warp2, persp_transform_mat_inv, (image.shape[1], image.shape[0])) 
# # Combine the result with the original image
# result = cv2.addWeighted(undistorted_image, 1, background, 0.4, 0)
# result2 = cv2.addWeighted(result, 1, background2, 0.7, 0)

# s_img = cv2.resize(f, (0,0), fx=0.2, fy=0.2) 
# x_offset=y_offset=10
# result2[y_offset:y_offset+s_img.shape[0], x_offset:x_offset+s_img.shape[1]] = s_img

# x_offset2, y_offset2=50 + s_img.shape[1], 10
# s_img2 = cv2.resize(p, (0,0), fx=0.2, fy=0.2)
# result2[y_offset2:y_offset2+s_img2.shape[0], x_offset2:x_offset2+s_img2.shape[1]] = s_img2

# x_offset3, y_offset3 = 90 + s_img.shape[1] + s_img2.shape[1], 10
# s_img3 = cv2.resize(searched[0], (0,0), fx=0.2, fy=0.2)
# result2[y_offset3:y_offset3+s_img3.shape[0], x_offset3:x_offset3+s_img3.shape[1]] = s_img3

# text = "Lane Curvatures:"
# text2 = "Left Lane = " + str("{:.7f}".format(rad[0]))
# text3 = "Right Lane = " + str("{:.7f}".format(rad[1]))

# result2 = cv2.putText(result2, text, (130 + s_img.shape[1] + s_img2.shape[1] + s_img3.shape[1], 30), cv2.FONT_HERSHEY_SIMPLEX,  
#                    0.85, (255, 255, 255) , 2, cv2.LINE_AA) 
# result2 = cv2.putText(result2, text2, (130 + s_img.shape[1] + s_img2.shape[1] + s_img3.shape[1], 90), cv2.FONT_HERSHEY_SIMPLEX,  
#                    0.85, (255, 255, 255) , 2, cv2.LINE_AA) 
# result2 = cv2.putText(result2, text3, (130 + s_img.shape[1] + s_img2.shape[1] + s_img3.shape[1], 150), cv2.FONT_HERSHEY_SIMPLEX,  
#                    0.85, (255, 255, 255) , 2, cv2.LINE_AA) 

# cv2.imshow('drawn', result2)
# cv2.waitKey(0)

# # cv2.imshow('drawn', warped_image)
# # cv2.waitKey(0)
# # print(thresholded_image)
# # cv2.imshow('drawn2', warped_image)
# # cv2.waitKey(0)
# # cv2.imshow('draw', searched[0])
# # cv2.waitKey(0)