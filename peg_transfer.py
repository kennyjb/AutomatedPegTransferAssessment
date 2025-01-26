import csv
import os

import cv2
import tensorflow as tf

from peg_transfer_fuzzy import *
from peg_transfer_ml import *

PATH_TO_SAVED_MODEL = "./model/"

# can be path to video or USB camera id
VIDEO_DIRECTORY = "./Videos/"
TOP_CAMERA = VIDEO_DIRECTORY + "Top.mp4"
FRONT_CAMERA = VIDEO_DIRECTORY + "Front.mp4"

OUTPUT_DIRECTORY = "./output/"
OUTPUT_TOP_VIDEO_FILENAME = OUTPUT_DIRECTORY + "top_output.mp4"
OUTPUT_FRONT_VIDEO_FILENAME = OUTPUT_DIRECTORY + "front_output.mp4"
OUTPUT_SPREADSHEET_FILENAME = OUTPUT_DIRECTORY + "output.csv"

FRAME_WIDTH_PIX = 1600
FRAME_HEIGHT_PIX = 1200
FRAME_WIDTH_MM = 210
FRAME_HEIGHT_MM = 170
FRAME_CENTER_MM = (FRAME_WIDTH_MM // 2, FRAME_HEIGHT_MM // 2)
FRAME_WIDTH_MM_PER_PIX = FRAME_WIDTH_MM / FRAME_WIDTH_PIX
FRAME_HEIGHT_MM_PER_PIX = FRAME_HEIGHT_MM / FRAME_HEIGHT_PIX

NUM_PEGS = 6

def calculate_bounding_box_center(box):
    box_tl = (int(box[1] * FRAME_WIDTH_PIX), int(box[0] * FRAME_HEIGHT_PIX))
    box_br = (int(box[3] * FRAME_WIDTH_PIX), int(box[2] * FRAME_HEIGHT_PIX))
    center_pix = ((box_tl[0] + box_br[0]) // 2, (box_tl[1] + box_br[1]) // 2)
    center_mm = (center_pix[0] * FRAME_WIDTH_MM_PER_PIX, center_pix[1] * FRAME_HEIGHT_MM_PER_PIX)

    return center_mm

def calculate_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def main():
    # load saved model
    detection_model = tf.saved_model.load(PATH_TO_SAVED_MODEL)
    print("Model successfully loaded!")

    # get fuzzy control systems
    Assessment_L_ctrl = get_left_grasper_fuzzy_controller()
    Assessment_R_ctrl = get_right_grasper_fuzzy_controller()
    Assessment_ctrl = get_final_fuzzy_controller()

    # load cameras
    top_camera = cv2.VideoCapture(TOP_CAMERA)
    front_camera = cv2.VideoCapture(FRONT_CAMERA)
    
    # get fps and ensure both cameras have the same fps
    fps = top_camera.get(cv2.CAP_PROP_FPS)
    assert(fps == front_camera.get(cv2.CAP_PROP_FPS))
    
    # ensure output directory exists
    os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)
    
    # open output video streams
    output_top_video = cv2.VideoWriter(OUTPUT_TOP_VIDEO_FILENAME, cv2.VideoWriter_fourcc(*'MP4V'), fps, (FRAME_WIDTH_PIX, FRAME_HEIGHT_PIX))
    output_front_video = cv2.VideoWriter(OUTPUT_FRONT_VIDEO_FILENAME, cv2.VideoWriter_fourcc(*'MP4V'), fps, (FRAME_WIDTH_PIX, FRAME_HEIGHT_PIX))

    # open output spreadsheet
    with open(OUTPUT_SPREADSHEET_FILENAME, 'w', newline='') as output_file:
        output_file_writer = csv.writer(output_file)
        # write header row
        output_file_writer.writerow(["Frame Number", 
                                     "Carry_L", "Carry_R", "Transfer", "Out_Field", "On_Peg_Cnt", "Drop_Cnt", 
                                     "Dis_Cam1_L", "Dis_Cam1_R", "H_Cam2_L", "H_Cam2_R", "LPA", "RPA", "FPA"])

        # process camera frames
        frame_number = 0
        while True:
            if not top_camera.grab() or not front_camera.grab():
                print("Reached end of video stream!")
                break

            frame_number += 1
            print(f"Processing frame {frame_number}...")

            # retrieve latest frames
            _, top_frame = top_camera.retrieve()
            _, front_frame = front_camera.retrieve()

            # run object detection on frames
            top_detections = run_object_detection(detection_model, top_frame)
            front_detections = run_object_detection(detection_model, front_frame)

            # check for left carry
            if len(top_detections[6]) > 0 or len(front_detections[6]) > 0:
                print("Left Carry")
                left_carry_flag = 1
            else:
                left_carry_flag = 0

            # check for right carry
            if len(top_detections[5]) > 0 or len(front_detections[5]) > 0:
                print("Right Carry")
                right_carry_flag = 1
            else:
                right_carry_flag = 0
            
            # check for transfer
            if len(top_detections[7]) > 0 or len(front_detections[7]) > 0:
                print("Transfer")
                transfer_flag = 1
            else:
                transfer_flag = 0
            
            # check for out-of-field
            if len(top_detections[10]) > 0 or len(front_detections[10]) > 0:
                print("Out-of-Field")
                out_field_flag = 1
            else:
                out_field_flag = 0
            
            # calculate on-peg and drop count
            # sort top camera detected pegs
            top_detected_pegs = [(0, x[0], x[1]) for x in top_detections[1]] + [(1, x[0], x[1]) for x in top_detections[2]]
            top_detected_pegs.sort(key=lambda x: x[1], reverse=True)
            top_detected_pegs = top_detected_pegs[0:NUM_PEGS]
            # sort front camera detected pegs
            front_detected_pegs = [(0, x[0], x[1]) for x in front_detections[1]] + [(1, x[0], x[1]) for x in front_detections[2]]
            front_detected_pegs.sort(key=lambda x: x[1], reverse=True)
            front_detected_pegs = front_detected_pegs[0:NUM_PEGS]
            # average camera peg counts
            detected_pegs = top_detected_pegs + front_detected_pegs
            on_peg_count = sum(1 for x in detected_pegs if x[0] == 0) // 2
            drop_count = sum(1 for x in detected_pegs if x[0] == 1) // 2
            print(f"On-Peg Count: {on_peg_count}")
            print(f"Drop Count: {drop_count}")

            # get left grasper center and height (if found)
            # check for Grasper_L in both frames
            if len(top_detections[4]) > 0 and len(front_detections[4]) > 0:
                top_left_grasper_pos = calculate_bounding_box_center(top_detections[4][0][1])
                front_left_grasper_pos = calculate_bounding_box_center(front_detections[4][0][1])
            else:
                top_left_grasper_pos = None
                front_left_grasper_pos = None

            # get right grasper center and height (if found)
            # check for Grasper_R in both frames
            if len(top_detections[3]) > 0 and len(front_detections[3]) > 0:
                top_right_grasper_pos = calculate_bounding_box_center(top_detections[3][0][1])
                front_right_grasper_pos = calculate_bounding_box_center(front_detections[3][0][1])
            else:
                top_right_grasper_pos = None
                front_right_grasper_pos = None

            # check if graspers found for fuzzy logic assessment
            if top_left_grasper_pos is not None and top_right_grasper_pos is not None and front_left_grasper_pos is not None and front_right_grasper_pos is not None:
                # top frame grasper center calculation (in mm)
                right_grasper_center_dist = calculate_distance(FRAME_CENTER_MM, top_right_grasper_pos)
                left_grasper_center_dist = calculate_distance(FRAME_CENTER_MM, top_left_grasper_pos)
                cv2.putText(top_frame, f"Right Grasper Center Distance: {right_grasper_center_dist:.2f}mm", (100, 900), cv2.FONT_HERSHEY_COMPLEX, 1, color=(255, 128, 0), thickness=2)
                cv2.putText(top_frame, f"Left Grasper Center Distance: {left_grasper_center_dist:.2f}mm", (100, 950), cv2.FONT_HERSHEY_COMPLEX, 1, color=(255, 128, 0), thickness=2)
                cv2.putText(top_frame, f"Right Grasper: (x={top_right_grasper_pos[0]:.2f}mm, y={top_right_grasper_pos[1]:.2f}mm)", (100, 1000), cv2.FONT_HERSHEY_COMPLEX, 1, color=(255, 128, 0), thickness=2)
                cv2.putText(top_frame, f"Left Grasper: (x={top_left_grasper_pos[0]:.2f}mm, y={top_left_grasper_pos[1]:.2f}mm)", (100, 1050), cv2.FONT_HERSHEY_COMPLEX, 1, color=(255, 128, 0), thickness=2)

                # front frame grasper center calculation (in mm)
                right_grasper_height = front_right_grasper_pos[1]
                left_grasper_height = front_left_grasper_pos[1]
                cv2.putText(front_frame, f"Right Grasper: (x={front_right_grasper_pos[0]:.2f}mm, y={front_right_grasper_pos[1]:.2f}mm)", (100, 1000), cv2.FONT_HERSHEY_COMPLEX, 1, color=(255, 128, 0), thickness=2)
                cv2.putText(front_frame, f"Left Grasper: (x={front_left_grasper_pos[0]:.2f}mm, y={front_left_grasper_pos[1]:.2f}mm)", (100, 1050), cv2.FONT_HERSHEY_COMPLEX, 1, color=(255, 128, 0), thickness=2)

                # infer fuzzy logic assessments
                left_assessment = infer_left_grasper_fuzzy_assessment(Assessment_L_ctrl, left_grasper_center_dist, left_grasper_height)
                right_assessment = infer_right_grasper_fuzzy_assessment(Assessment_R_ctrl, right_grasper_center_dist, right_grasper_height)
                final_assessment = infer_final_fuzzy_assessment(Assessment_ctrl, right_assessment, left_assessment)
                print(f"Left Performance Assessment: {left_assessment:.2f}%")
                print(f"Right Performance Assessment: {right_assessment:.2f}%")
                print(f"Final Performance Assessment: {final_assessment:.2f}%")
            else:
                left_grasper_center_dist = -1
                right_grasper_center_dist = -1
                left_grasper_height = -1
                right_grasper_height = -1
                left_assessment = -1
                right_assessment = -1
                final_assessment = -1

            # write results to spreadsheet
            output_file_writer.writerow([frame_number, 
                                         left_carry_flag, right_carry_flag, transfer_flag, out_field_flag,
                                         on_peg_count, drop_count,
                                         left_grasper_center_dist, right_grasper_center_dist, 
                                         left_grasper_height, right_grasper_height, 
                                         left_assessment, right_assessment, final_assessment])

            # display the frames
            cv2.namedWindow("Top Camera", cv2.WINDOW_KEEPRATIO)
            cv2.imshow("Top Camera", top_frame)
            cv2.namedWindow("Front Camera", cv2.WINDOW_KEEPRATIO)
            cv2.imshow("Front Camera", front_frame)

            # save the frames
            output_top_video.write(top_frame)
            output_front_video.write(front_frame)

            # check for a key press
            key = cv2.waitKey(1) & 0xFF
            if key == ord("d"): # done key pressed
                break
    
    # release cameras and destroy windows
    top_camera.release()
    front_camera.release()
    output_top_video.release()
    output_front_video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
