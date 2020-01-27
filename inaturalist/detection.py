"""
This notebook will demontrate a pre-trained model to recognition plate number in an image.
Make sure to follow the [installation instructions](https://github.com/imamdigmi/plate-number-recognition#setup) before you start.
"""
import os
import sys
import tarfile
import zipfile
from collections import defaultdict
from io import StringIO

import cv2 as cv2
import humanfriendly
import numpy as np
import six.moves.urllib as urllib
import tensorflow as tf
from matplotlib import pyplot as plt
import time

# Object detection imports
# Here are the imports from the object detection module.

from PIL import Image

# NUM_PARALLEL_EXEC_UNITS = 6

# os.environ["OMP_NUM_THREADS"] = str(NUM_PARALLEL_EXEC_UNITS)
# os.environ["KMP_BLOCKTIME"] = "30"
# os.environ["KMP_SETTINGS"] = "1"
# os.environ["KMP_AFFINITY"]= "granularity=fine,verbose,compact,1,0"


sys.path.append("/home/gaiar/developer/models/research/")
sys.path.append("/home/gaiar/developer/models/research/slim")

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

FILE_OUTPUT = "heron.avi"

# Checks and deletes the output file
# You cant have a existing file or it will through an error
if os.path.isfile(FILE_OUTPUT):
    os.remove(FILE_OUTPUT)

# Playing video from file
cap = cv2.VideoCapture("heron.mp4")

# Default resolutions of the frame are obtained.The default resolutions are system dependent.
# We convert the resolutions from float to integer.
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# Define the codec and create VideoWriter object.The output is stored in 'output.avi' file.
out = cv2.VideoWriter(
    FILE_OUTPUT,
    cv2.VideoWriter_fourcc("M", "J", "P", "G"),
    10,
    (frame_width, frame_height),
)

sys.path.append("..")


# Model preparation
MODEL_NAME = "faster_rcnn_resnet101_fgvc_2018_07_19"
PATH_TO_CKPT = MODEL_NAME + "/frozen_inference_graph.pb"
PATH_TO_LABELS = os.path.join("data", "fgvc_2854_classes_label_map.pbtxt")
NUM_CLASSES = 2854

# Load a (frozen) Tensorflow model into memory.
print("Loading model...")
start_time = time.time()

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, "rb") as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name="")
elapsed = time.time() - start_time
print("Loaded model in {}".format(humanfriendly.format_timespan(elapsed)))

# Loading label map
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(
    label_map, max_num_classes=NUM_CLASSES, use_display_name=True
)
category_index = label_map_util.create_category_index(categories)


# config = tf.ConfigProto(intra_op_parallelism_threads = NUM_PARALLEL_EXEC_UNITS,
#         inter_op_parallelism_threads = 1,
#         allow_soft_placement = True,
#         device_count = {'CPU': NUM_PARALLEL_EXEC_UNITS })

def load_img(paths, grayscale=False, target_size=None, crop_size=None,
             interp=None):
    assert cv2 is not None, '`load_img` requires `cv2`.'
    if interp is None:
        interp = cv2.INTER_CUBIC
    if not isinstance(paths, list):
        paths = [paths]
    if len(paths) > 1 and (target_size is None or
                           isinstance(target_size, int)):
        raise ValueError('A tuple `target_size` should be provided '
                         'when loading multiple images.')

    def _load_img(path):
        img = cv2.imread(path)
        if target_size:
            if isinstance(target_size, int):
                hw_tuple = tuple([x * target_size // min(img.shape[:2])
                                  for x in img.shape[1::-1]])
            else:
                hw_tuple = (target_size[1], target_size[0])
            if img.shape[1::-1] != hw_tuple:
                img = cv2.resize(img, hw_tuple, interpolation=interp)
        img = img[:, :, ::-1]
        if len(img.shape) == 2:
            img = np.expand_dims(img, -1)
        return img

    if len(paths) > 1:
        imgs = np.zeros((len(paths),) + target_size + (3,), dtype=np.float32)
        for (i, path) in enumerate(paths):
            imgs[i] = _load_img(path)
    else:
        imgs = np.array([_load_img(paths[0])], dtype=np.float32)

    return imgs

with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
        # Definite input and output Tensors for detection_graph
        image_tensor = detection_graph.get_tensor_by_name("image_tensor:0")

        # Each box represents a part of the image where a particular object was detected.
        detection_boxes = detection_graph.get_tensor_by_name("detection_boxes:0")

        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        detection_scores = detection_graph.get_tensor_by_name("detection_scores:0")
        detection_classes = detection_graph.get_tensor_by_name("detection_classes:0")
        num_detections = detection_graph.get_tensor_by_name("num_detections:0")

        while cap.isOpened():
            # Capture frame-by-frame
            ret, frame = cap.read()
            inp = cv2.resize(frame, (600, 600))
            # inp = inp[:, :, [2, 1, 0]]  # BGR2RGB
            inp = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB)
            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_np_expanded = np.expand_dims(inp, axis=0)

            # Actual detection.

            start_time = time.time()
            (boxes, scores, classes, num) = sess.run(
                [detection_boxes, detection_scores, detection_classes, num_detections],
                feed_dict={image_tensor: image_np_expanded},
            )
            elapsed = time.time() - start_time
            print("Inference took {}".format(humanfriendly.format_timespan(elapsed)))

            # Visualization of the results of a detection.
            vis_util.visualize_boxes_and_labels_on_image_array(
                frame,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=8,
            )

            if ret == True:
                # Saves for video
                out.write(frame)

                # # Display the resulting frame
                # cv2.imshow("Charving Detection", frame)

                # # Close window when "Q" button pressed
                # if cv2.waitKey(1) & 0xFF == ord("q"):
                #     break
            else:
                break

    # When everything done, release the video capture and video write objects
    cap.release()
    out.release()

    # Closes all the frames
    cv2.destroyAllWindows()
