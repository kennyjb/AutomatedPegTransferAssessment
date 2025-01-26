import csv
import serial
import time

import cv2
import numpy as np

PORT = "COM4"
BAUDRATE = 115200
DATA_BITS = 8
STOP_BITS = 1
PARITY = 'N'
TIMEOUT = 1 # seconds

INPUT_SPREADSHEET = "./output/output.csv"
FPS = 30

OUTPUT_WINDOW_SIZE = (600, 320)

def send_and_receive(connection, message):
    connection.write(message.encode())
    response = connection.readline().decode()
    return response

def display_response(status, completed_lr_cnt, completed_rl_cnt, seconds_elapsed, left_assessment, right_assessment, final_assessment):
    # create black image
    image = np.zeros((OUTPUT_WINDOW_SIZE[1], OUTPUT_WINDOW_SIZE[0], 3))
    if (status & 0x004) > 1:
        status_message = "Error: Time Up"
    elif (status & 0x008) > 1:
        status_message = "Error: Out-of-Field"
    elif (status & 0x010) > 1:
        status_message = "Error: Drop Count > 1"
    elif (status & 0x020) > 1:
        status_message = "Error: Drop Count = 1"
    elif (status & 0x040) > 1:
        status_message = "L->R Carry Left Active"
    elif (status & 0x080) > 1:
        status_message = "L->R Transfer Active"
    elif (status & 0x100) > 1:
        status_message = "L->R Carry Right Active"
    elif (status & 0x200) > 1:
        status_message = "R->L Carry Right Active"
    elif (status & 0x400) > 1:
        status_message = "R->L Transfer Active"
    elif (status & 0x800) > 1:
        status_message = "R->L Carry Left Active"
    elif (status & 0x002) > 1:
        status_message = "Active"
    elif (status & 0x001) > 1:
        status_message = "Finished Successfully"
    else:
        status_message = "Unknown"
    # display response
    cv2.putText(image, f"Status: {status_message}", (20, 40), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0))
    cv2.putText(image, f"Completed L->R Count: {completed_lr_cnt}", (20, 80), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0))
    cv2.putText(image, f"Completed R->L Count: {completed_rl_cnt}", (20, 120), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0))
    cv2.putText(image, f"Time Elapsed: {seconds_elapsed} seconds", (20, 160), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0))
    cv2.putText(image, f"Left Assessment: {left_assessment}%", (20, 200), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0))
    cv2.putText(image, f"Right Assessment: {right_assessment}%", (20, 240), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0))
    cv2.putText(image, f"Final Assessment: {final_assessment}%", (20, 280), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0))
    # show results
    cv2.imshow("Results", image)
    # poll for user input
    cv2.waitKey(1)

def main():
    with serial.Serial(port=PORT, baudrate=BAUDRATE, bytesize=DATA_BITS, stopbits=STOP_BITS, parity=PARITY, timeout=TIMEOUT) as connection:
        # reset peg transfer
        print(send_and_receive(connection, "reset\n"))
        # start peg transfer
        print(send_and_receive(connection, "start\n"))
        num_frames = 0
        with open(INPUT_SPREADSHEET, 'r') as csvfile:
            csvreader = csv.reader(csvfile)
            # skip header row
            next(csvreader)

            # record start time
            start_time = time.time()

            # iterate through frames
            for line in csvreader:
                num_frames += 1

                # read frame data from spreadsheet
                carry_l = int(line[1]) & 0x1
                carry_r = int(line[2]) & 0x1
                transfer = int(line[3]) & 0x1
                out_of_field = int(line[4]) & 0x1
                on_peg_cnt = int(line[5]) & 0xF
                drop_cnt = int(line[6]) & 0xF
                left_distance = round(float(line[7])) & 0xFF
                left_height = round(float(line[9])) & 0xFF
                right_distance = round(float(line[8])) & 0xFF
                right_height = round(float(line[10])) & 0xFF

                # send update command
                flags = (out_of_field << 3) | (transfer << 2) | (carry_r << 1) | (carry_l << 0)
                command = f"update {flags:01x} {on_peg_cnt:01x} {drop_cnt:01x} {left_distance:02x} {left_height:02x} {right_distance:02x} {right_height:02x}\n"
                response = send_and_receive(connection, command).split()
                if len(response) == 8:
                    status = int(response[1], 16)
                    completed_lr_cnt = int(response[2], 16)
                    completed_rl_cnt = int(response[3], 16)
                    seconds_elapsed = int(response[4], 16)
                    left_assessment = int(response[5], 16)
                    right_assessment = int(response[6], 16)
                    final_assessment = int(response[7], 16)

                    display_response(status, completed_lr_cnt, completed_rl_cnt, seconds_elapsed, left_assessment, right_assessment, final_assessment)

                # record end time
                total_time = time.time() - start_time
                actual_time = num_frames / FPS
                # delay for framerate
                delay = actual_time - total_time
                if delay > 0:
                    time.sleep(delay)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()