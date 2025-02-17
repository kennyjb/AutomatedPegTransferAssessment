import cv2
import numpy as np

INPUT_TOP_VIDEO_FILENAME = "./output/top_output.mp4"
INPUT_FRONT_VIDEO_FILENAME = "./output/front_output.mp4"
INPUT_ACCELERATOR_VIDEO_FILENAME = "./output/accelerator_output.mp4"
OUTPUT_VIDEO_FILENAME = "./output/final_output.mp4"

CAMERA_FRAME_SIZE = (1600, 1200)
ACCELERATOR_VIDEO_FRAME_SIZE = (600, 320)
OUTPUT_FRAME_SIZE = (1600, 920)
FPS = 30

def main():
    top_video = cv2.VideoCapture(INPUT_TOP_VIDEO_FILENAME)
    front_video = cv2.VideoCapture(INPUT_FRONT_VIDEO_FILENAME)
    accelerator_video = cv2.VideoCapture(INPUT_ACCELERATOR_VIDEO_FILENAME)
    output_video = cv2.VideoWriter(OUTPUT_VIDEO_FILENAME, cv2.VideoWriter_fourcc(*'MP4V'), FPS, OUTPUT_FRAME_SIZE)
    
    while True:
        if not top_video.grab() or not front_video.grab() or not accelerator_video.grab():
            print("Reached end of video stream!")
            break

        # retrieve latest frames
        _, top_frame = top_video.retrieve()
        _, front_frame = front_video.retrieve()
        _, accelerator_frame = accelerator_video.retrieve()

        # create combined frame
        top_frame = cv2.resize(top_frame, None, fx=0.5, fy=0.5)
        front_frame = cv2.resize(front_frame, None, fx=0.5, fy=0.5)
        output_frame = np.concatenate((top_frame, front_frame), axis=1)
        (height, width, _) = np.shape(accelerator_frame)
        new_accelerator_frame = np.zeros((ACCELERATOR_VIDEO_FRAME_SIZE[1], OUTPUT_FRAME_SIZE[0], 3), dtype=np.uint8)
        new_accelerator_frame[0:height, 0:width, :] = accelerator_frame
        output_frame = np.concatenate((output_frame, new_accelerator_frame), axis=0)

        # save the frame
        output_video.write(output_frame)

        # check for a key press
        key = cv2.waitKey(1) & 0xFF
        if key == ord("d"): # done key pressed
            break
    
    # release cameras and destroy windows
    top_video.release()
    front_video.release()
    accelerator_video.release()
    output_video.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
