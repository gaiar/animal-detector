import os
import sys
import time
from pathlib import Path, PurePath
from pprint import pprint
from statistics import mean

import cv2 as cv
import humanfriendly
import numpy as np
import tensorflow as tf

THREASHOLD = 0.5


def get_output_file(input_file):
    return Path(PurePath(input_file).stem + ".avi")


def image_resize(image, width=None, height=None, inter=cv.INTER_LANCZOS4):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized


clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

wb = cv.xphoto.createSimpleWB()
wb.setP(0.3)

FILE_INPUT = "squirrel_2.mp4"
FILE_OUTPUT = get_output_file(FILE_INPUT)

# Checks and deletes the output file
# You cant have a existing file or it will through an error
if os.path.isfile(FILE_OUTPUT):
    os.remove(FILE_OUTPUT)

# Playing video from file
cap = cv.VideoCapture(FILE_INPUT)

# Default resolutions of the frame are obtained.The default resolutions are system dependent.
# We convert the resolutions from float to integer.
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# Define the codec and create VideoWriter object.The output is stored in 'output.avi' file.
outv = cv.VideoWriter(
    str(FILE_OUTPUT),
    cv.VideoWriter_fourcc("M", "J", "P", "G"),
    10,
    (frame_width, frame_height),
)

sys.path.append("..")

classes = {}
bbox_categories = [
    {"id": 0, "name": "empty"},
    {"id": 1, "name": "animal"},
    {"id": 2, "name": "person"},
    {"id": 3, "name": "group"},  # group of animals
    {"id": 4, "name": "vehicle"},
]
detections = {}
detections["classes"] = []
detections["scores"] = []
detections["boxes"] = []
detections["numbers"] = []

# boxes.append(box)
# scores.append(score)
# classes.append(clss)


for cat in bbox_categories:
    classes[int(cat["id"])] = cat["name"]


def load_model(checkpoint):
    """
    Load a detection model (i.e., create a graph) from a .pb file
    """

    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(checkpoint, "rb") as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name="")

    return detection_graph


def enchance_image(frame):
    temp_img = frame
    img_wb = wb.balanceWhite(temp_img)

    img_lab = cv.cvtColor(img_wb, cv.COLOR_BGR2Lab)

    l, a, b = cv.split(img_lab)
    img_l = clahe.apply(l)
    img_clahe = cv.merge((img_l, a, b))

    frame = cv.cvtColor(img_clahe, cv.COLOR_Lab2BGR)
    return frame


def check_detections(preds):
    return None


def postprocess(frame, classId, score, bbox):
    if score > THREASHOLD:
        left = bbox[1] * cols
        top = bbox[0] * rows
        right = bbox[3] * cols
        bottom = bbox[2] * rows
        cv.rectangle(
            frame,
            (int(left), int(top)),
            (int(right), int(bottom)),
            (125, 255, 51),
            thickness=2,
        )
        label = "%.2f" % score
        if classes:
            assert classId < len(classes)
            label = "%s:%s" % (classes[classId], label)
        labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        top = max(top, labelSize[1])
        cv.rectangle(
            frame,
            (int(left), int(top - round(1.5 * labelSize[1]))),
            (int(left + round(1.5 * labelSize[0])), int(top + baseLine)),
            (255, 255, 255),
            cv.FILLED,
        )
        cv.putText(
            frame,
            label,
            (int(left), int(top)),
            cv.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 0, 0),
            1,
        )
    return frame


def draw_predictions(classId, conf, left, top, right, bottom):
    pass


def write_raw_video(frames):
    pass


def calculate_stats(n_frames, detections):
    print("[INFO] :: Video length {} frames".format(n_frames))
    avg_score = []
    #[item for sublist in l for item in sublist]
    avg_score = [score for subscore in detections["scores"] for score in subscore if score > THREASHOLD]
    print("[INFO] :: Average score is {}".format(mean(avg_score)))
    avg_detect = sum(detections["numbers"])
    print("[INFO] :: Number of detections is {}".format(avg_detect))
    print("[INFO] :: Average detections on frame is {0}".format(avg_detect/n_frames))


detection_graph = None
# Load and run detector on target images
print("Loading model...")
start_time = time.time()
if detection_graph is None:
    detection_graph = load_model("megadetector_v3.pb")
elapsed = time.time() - start_time
print("Loaded model in {}".format(humanfriendly.format_timespan(elapsed)))


# exit()





with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:

        n_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
        
        i = 0
        while cap.isOpened():
            # Capture frame-by-frame
            ret, img = cap.read()
            

            # Read and preprocess an image.
            # img = cv.imread("rackoon_2.png")

            if img is not None:

                try:
                   img = enchance_image(img)
                except Exception as e:
                   print("[ERROR] :: " + e)
                rows = img.shape[0]
                cols = img.shape[1]
                inp = cv.resize(img, (300, 300))
                inp = inp[:, :, [2, 1, 0]]  # BGR2RGB


                # # Actual detection
                # (box, score, clss, num_detections) = sess.run(
                #         [box, score, clss, num_detections],
                #         feed_dict={image_tensor: imageNP_expanded})

                # boxes.append(box)
                # scores.append(score)
                # classes.append(clss)

                # Run the model
                out = sess.run(
                    [
                        sess.graph.get_tensor_by_name("num_detections:0"),
                        sess.graph.get_tensor_by_name("detection_scores:0"),
                        sess.graph.get_tensor_by_name("detection_boxes:0"),
                        sess.graph.get_tensor_by_name("detection_classes:0"),
                    ],
                    feed_dict={
                        "image_tensor:0": inp.reshape(1, inp.shape[0], inp.shape[1], 3)
                    },
                )

                # Visualize detected bounding boxes.
                # pprint(out)
                num_detections = int(out[0][0])

                detections["scores"].append(out[1][0])
                detections["classes"].append(out[3][0])
                detections["boxes"].append(out[2][0])
                detections["numbers"].append(num_detections)

                for i in range(num_detections):
                    classId = int(out[3][0][i])
                    score = float(out[1][0][i])
                    bbox = [float(v) for v in out[2][0][i]]

                    img = postprocess(img, classId, score, bbox)
           
            if ret == True:
                # Saves for video
                # img = image_resize(img, width=800)
                # img = enchance_image(img)

                outv.write(img)

                # Display the resulting frame
                cv.imshow("Charving Detection", img)
                i+=1
                # Close window when "Q" button pressed
                if cv.waitKey(1) & 0xFF == ord("q"):
                    break
                
            else:
                print("File ended")
                
                calculate_stats(n_frames,detections)
                break

        # cv.rectangle(frame, (left, top), (right, bottom), (255, 178, 50), 3)

        # Get the label for the class name and its confidence

        # Display the label at the top of the bounding box

        cap.release()
        outv.release()


# if __name__ == "__main__":
#     main()
