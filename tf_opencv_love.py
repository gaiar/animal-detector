import argparse
import glob
import os
import sys
import time
from datetime import datetime
from pathlib import Path, PurePath
from pprint import pprint
from statistics import mean

import cv2 as cv
import humanfriendly
import numpy as np
import tensorflow as tf
from tqdm import tqdm, trange
import gc
import shutil

from filevideostream import FileVideoStream


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 1.0

# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
# tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

DEFAULT_CONFIDENCE_THRESHOLD = 0.3
BS = 2
DETECTION_FILENAME_INSERT = "_detections"
DISPLAY_RESULTS = False

# Don't enable. Makes things worse.
ENABLE_ENCHANCER = False


def get_output_file(out_dir, input_file):
    return Path(
        out_dir,
        PurePath(input_file).stem + datetime.now().strftime("_%H_%M_%d_%m_%Y") + ".mp4",
    )


def move_input_file(input_file):
    new_location = Path(
        "processed/",
        PurePath(input_file).stem + datetime.now().strftime("_%H_%M_%d_%m_%Y") + ".mp4",
    )
    shutil.move(input_file, new_location)


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

    return cv.resize(image, dim, interpolation=inter)


sys.path.append("..")

image_extensions = [".jpg", ".jpeg", ".gif", ".png"]
video_extensions = [".mp4", ".mov", ".avi", ".mkv"]

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
detections["frames"] = []

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

    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    wb = cv.xphoto.createSimpleWB()
    wb.setP(0.3)

    temp_img = frame
    img_wb = wb.balanceWhite(temp_img)
    img_lab = cv.cvtColor(img_wb, cv.COLOR_BGR2Lab)
    l, a, b = cv.split(img_lab)
    img_l = clahe.apply(l)
    img_clahe = cv.merge((img_l, a, b))
    return cv.cvtColor(img_clahe, cv.COLOR_Lab2BGR)


def check_detections(preds):
    return None


def postprocess_all(detections, n_frames, out_video):
    print(
        "[INFO] :: Going through {0} detections and {1} frames".format(
            len(detections["frames"]), n_frames
        )
    )
    print(
        "[DEBUG] :: Length Classes: {0}, Length Scores: {1}, Length Boxes: {2}, Length Frames: {3}".format(
            len(detections["classes"]),
            len(detections["scores"]),
            len(detections["boxes"]),
            len(detections["frames"]),
        ),
    )
    n_frames = len(detections["frames"])
    for i in tqdm(range(n_frames)):
        classes = detections["classes"][i]
        scores = detections["scores"][i]
        bboxes = detections["boxes"][i]
        num_detections = detections["numbers"][i]
        frame = detections["frames"][i]

        for j in range(num_detections):
            class_id = int(classes[j])
            score = float(scores[j])
            bbox = [float(v) for v in bboxes[j]]
            frame = postprocess(frame, class_id, score, bbox)
        # print("[DEBUG] :: Writing to file frame {0}".format(i))
        # print("Writing frame {0}".format(frame.shape))
        out_video.write(frame)

        if DISPLAY_RESULTS:
            # Display the resulting frame
            cv.imshow("Animals Detection. Frames number {0}".format(n_frames), frame)
            # Close window when "Q" button pressed
            if cv.waitKey(1) & 0xFF == ord("q"):
                print("[WARN] :: Exiting on button Q pressed")
                break
    del detections
    gc.collect()


def run_inference_on_video(fvs, video_file, sess):

    n_frames = fvs.n_frames

    detections = {}
    detections["classes"] = []
    detections["scores"] = []
    detections["boxes"] = []
    detections["numbers"] = []
    detections["frames"] = []
    f = 0

    check_done = False
    while fvs.running():
        frames = fvs.get_batch(BS)

        print(
            "[INFO] :: Detecting frame {0} out of {1}. Progress {2}%".format(
                f, n_frames, round(f / n_frames * 100)
            )
        )

        image_np_list = []
        for frame in frames:
            inp = cv.resize(frame, (300, 300))
            inp = inp[:, :, [2, 1, 0]]  # BGR2RGB      
            #image_np_list.append(inp)
            image_np_list.append(np.asarray(inp, np.uint8))
        image_np_expanded = np.asarray(image_np_list)

        # Run the model
        batch_start_time = time.time()

        outs = sess.run(
            [
                sess.graph.get_tensor_by_name("num_detections:0"),
                sess.graph.get_tensor_by_name("detection_scores:0"),
                sess.graph.get_tensor_by_name("detection_boxes:0"),
                sess.graph.get_tensor_by_name("detection_classes:0"),
            ],
            feed_dict={"image_tensor:0": image_np_expanded},
        )

        elapsed_batch_time = time.time() - batch_start_time
        print(
            "[INFO] :: One batch detection took {0}".format(
                humanfriendly.format_timespan(elapsed_batch_time)
            )
        )

        for det in range(len(frames)):
            try:
                num_detections = int(outs[0][det])

                detections["scores"].append(outs[1][det])
                detections["classes"].append(outs[3][det])
                detections["boxes"].append(outs[2][det])
                detections["numbers"].append(num_detections)
                detections["frames"].append(frames[det])
            except Exception as e:
                print("No more frames. {0}".format(e))
                break

        if f >= round(n_frames // 2) and not check_done:
            check_done = True
            _, temp_avg_detect = calculate_stats(n_frames, detections)
            print(
                "[INFO] :: Intermediate results. {0} detections at half of the video".format(
                    temp_avg_detect
                )
            )
            if not temp_avg_detect > 1:
                print("[WARN] :: Nothing found at half of the video. Stopping")
                break

        f += BS

    print("[INFO] :: File {0} ended".format(video_file))

    return detections


def postprocess(frame, class_id, score, bbox):
    # frame = image_resize(frame, width=800)
    # frame = enchance_image(frame)
    rows = frame.shape[0]
    cols = frame.shape[1]
    if score > DEFAULT_CONFIDENCE_THRESHOLD:
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
            assert class_id < len(classes)
            label = "%s:%s" % (classes[class_id], label)
            # print("[DEBUG] :: Found {0}".format(label))
        label_size, base_line = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        top = max(top, label_size[1])
        cv.rectangle(
            frame,
            (int(left), int(top - round(1.5 * label_size[1]))),
            (int(left + round(1.5 * label_size[0])), int(top + base_line)),
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


def calculate_stats(n_frames, detections):
    print("[INFO] :: Video length {} frames".format(n_frames))
    avg_score = []
    # [item for sublist in l for item in sublist]
    avg_score = [
        score
        for subscore in detections["scores"]
        for score in subscore
        if score > DEFAULT_CONFIDENCE_THRESHOLD
    ]
    if len(avg_score) > 0:
        print("[INFO] :: Average score is {}".format(mean(avg_score)))
    else:
        print("[INFO] :: Average score is {}".format(len(avg_score)))

    avg_detect = len(avg_score)
    print("[INFO] :: Number of meaningful detections is {}".format(avg_detect))
    print("[INFO] :: Average detections per frame is {0}".format(avg_detect / n_frames))
    return avg_score, avg_detect


def is_image_file(s):
    """
    Check a file's extension against a hard-coded set of image file extensions    '
    """
    ext = os.path.splitext(s)[1]
    return ext.lower() in image_extensions


def is_video_file(s):
    """
    Check a file's extension against a hard-coded set of image file extensions    '
    """
    ext = os.path.splitext(s)[1]
    return ext.lower() in video_extensions


def find_video_strings(strings):
    """
    Given a list of strings that are potentially image file names, look for strings
    that actually look like image file names (based on extension).
    """
    files = []
    for item in strings:
        video_file = Path(item)
        if video_file.exists() and is_video_file(video_file):
            files.append(video_file)
    return files

def find_videos(dir_name, recursive=False):
    if recursive:
        strings = glob.glob(os.path.join(dir_name, "**", "*.*"), recursive=True)
    else:
        strings = glob.glob(os.path.join(dir_name, "*.*"))

    return find_video_strings(strings)


def load_and_run_detector(
    model_file, video_file_names, confidence_threshold, output_dir,
):
    # model_file = "megadetector_v3.pb"
    if not model_file:
        model_file = "megadetector/exported_model/frozen_inference_graph_optimized.pb"
    video_file_names = video_file_names

    # Load and run detector on target images
    print("[DEBUG] :: Loading model {0}".format(model_file))

    with tf.Session(graph=tf.Graph(), config=config) as sess:
        # graph = tf.get_default_graph()
        start_time = time.time()
        tf.saved_model.loader.load(sess, ["serve"], model_file)
        elapsed = time.time() - start_time
        print("Loaded model in {}".format(humanfriendly.format_timespan(elapsed)))

        # vfiles = tqdm(video_file_names)
        for video_file in video_file_names:
            # vfiles.set_description("Processing file {0}".format(str(video_file)))

            fvs = FileVideoStream(video_file).start()
            time.sleep(1.0)

            n_frames = fvs.n_frames
            start_time = time.time()

            detections = run_inference_on_video(fvs, video_file, sess)

            elapsed = time.time() - start_time
            print(
                "[INFO] :: Detection took {}".format(
                    humanfriendly.format_timespan(elapsed)
                )
            )
            _, avg_detect = calculate_stats(n_frames, detections)
            if avg_detect > 5:
                # TODO: Pass to function all stuff about VideoWriter
                out_video_file = get_output_file(output_dir, video_file)
                frame_width = fvs.frame_width
                frame_height = fvs.frame_height
                fps = fvs.frame_rate
                # fourcc = cv.VideoWriter_fourcc(*"hvc1")
                fourcc = cv.VideoWriter_fourcc(*"mp4v")
                out_video = cv.VideoWriter(
                    str(out_video_file), fourcc, fps, (frame_width, frame_height),
                )
                postprocess_all(detections, n_frames, out_video)
                out_video.release()
            else:
                print("[WARN] :: Nothing meaningful found on video")

            fvs.stop()
            move_input_file(video_file)
            del detections
            # del frames
            gc.collect()

            # break


def main():

    # python run_tf_detector.py "D:\temp\models\object_detection\megadetector\megadetector_v2.pb" --video "D:\temp\demo_images\test\S1_J08_R1_PICT0120.JPG"
    # python run_tf_detector.py "D:\temp\models\object_detection\megadetector\megadetector_v2.pb" --videos "d:\temp\demo_images\test"
    # python run_tf_detector.py "d:\temp\models\megadetector_v3.pb" --videos "d:\temp\test\in" --outputDir "d:\temp\test\out"

    parser = argparse.ArgumentParser()
    parser.add_argument("detector_file", type=str)
    parser.add_argument(
        "--videos",
        action="store",
        type=str,
        default="",
        help="Directory to search for videos, with optional recursion",
    )
    parser.add_argument(
        "--video",
        action="store",
        type=str,
        default="",
        help="Single file to process, mutually exclusive with videos",
    )
    parser.add_argument(
        "--threshold",
        action="store",
        type=float,
        default=DEFAULT_CONFIDENCE_THRESHOLD,
        help="Confidence threshold, don't render boxes below this confidence",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recurse into directories, only meaningful if using --videos",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU detection, even if a GPU is available",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Directory for output videos (defaults to same as input)",
    )

    if len(sys.argv[1:]) == 0:
        parser.print_help()
        parser.exit()

    args = parser.parse_args()

    if len(args.video) > 0 and len(args.videos) > 0:
        raise Exception("Cannot specify both video file and videos dir")
    elif len(args.video) == 0 and len(args.videos) == 0:
        raise Exception("Must specify either an video file or an videos directory")

    if len(args.video) > 0:
        video_file_names = [args.video]
    else:
        video_file_names = find_videos(args.videos, args.recursive)

    if args.cpu:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    # Hack to avoid running on already-detected images
    # print(video_file_names)
    # video_file_names = [
    #     x for x in video_file_names if DETECTION_FILENAME_INSERT not in x
    # ]

    print("Running detector on {} videos".format(len(video_file_names)))

    load_and_run_detector(
        model_file=args.detector_file,
        video_file_names=video_file_names,
        confidence_threshold=args.threshold,
        output_dir=args.output,
    )


if __name__ == "__main__":
    main()
