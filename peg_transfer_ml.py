import cv2
import numpy as np
import tensorflow as tf

IDS_TO_LABELS = {
    1: "OnPeg",
    2: "OutPeg",
    3: "Grasper_R",
    4: "Grasper_L",
    5: "Carry_R",
    6: "Carry_L",
    7: "Transfer",
    8: "Pick_R",
    9: "Pick_L",
    10: "OutField"
}

MIN_SCORES = {
    1: 0.4,
    2: 0.9,
    3: 0.4,
    4: 0.4,
    5: 0.4,
    6: 0.4,
    7: 0.4,
    8: 0.4,
    9: 0.4,
    10: 0.9
}

MAX_ITEMS_TO_KEEP = {
    1: 6,
    2: 6,
    3: 1,
    4: 1,
    5: 1,
    6: 1,
    7: 1,
    8: 1,
    9: 1,
    10: 1
}

def run_inference_for_frame(model, image):
    image = np.asarray(image)
    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis,...]

    # Run inference
    model_fn = model.signatures['serving_default']
    output_dict = model_fn(input_tensor)

    # All outputs are batched tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(output_dict.pop('num_detections'))
    output_dict = {key:value[0,:num_detections].numpy() for key,value in output_dict.items()}
    output_dict['num_detections'] = num_detections
    # detection_classes should be ints
    output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)

    return output_dict

def process_detections(detections):
    classes = detections["detection_classes"]
    scores = detections["detection_scores"]
    boxes = detections["detection_boxes"]

    processed_detections = {key:[] for key in MIN_SCORES}
    for i in range(classes.shape[0]):
        # check if score is higher than threshold for predicted class
        if scores[i] > MIN_SCORES[classes[i]]:
            processed_detections[classes[i]].append((scores[i], boxes[i]))
    
    # sort each detection category from highest to lowest score
    # keep specified number of items per class
    for key in processed_detections:
        processed_detections[key].sort(key=lambda x: x[0], reverse=True)
        processed_detections[key] = processed_detections[key][0:MAX_ITEMS_TO_KEEP[key]]
    
    return processed_detections

def visualize_detections(image, detections):
    (frame_height, frame_width) = image.shape[0:2]

    # visualize detections
    for key in detections:
        for item in detections[key]:
            detected_class = key
            detected_score = item[0]
            detected_box = item[1]

            box_tl = (int(detected_box[1] * frame_width), int(detected_box[0] * frame_height))
            box_br = (int(detected_box[3] * frame_width), int(detected_box[2] * frame_height))
            cv2.rectangle(image, box_tl, box_br, color=(0, 255, 0), thickness=2)

            text = f"{IDS_TO_LABELS[detected_class]} | {detected_score:.0%}"
            text_bl = (box_tl[0], box_tl[1] - 10)
            cv2.putText(image, text, text_bl, cv2.FONT_HERSHEY_COMPLEX, 1, color=(0, 255, 0), thickness=2)

def run_object_detection(model, image):
    # infer, process, and visualize detections
    model_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    detections = run_inference_for_frame(model, model_image)
    detections = process_detections(detections)
    visualize_detections(image, detections)

    return detections
