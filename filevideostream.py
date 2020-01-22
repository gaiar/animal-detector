# import the necessary packages
from threading import Thread
import sys
import cv2
import time
from queue import Queue


class FileVideoStream:
    def __init__(self, path, transform=None, resize=None, queue_size=128):
        # initialize the file video stream along with the boolean
        # used to indicate if the thread should be stopped or not
        self.stream = cv2.VideoCapture(str(path))
        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, queue_size)
        self.n_frames = int(self.stream.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fw = self.stream.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.fh = self.stream.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.fps = self.stream.get(cv2.CAP_PROP_FPS)
        self.stopped = False
        self.transform = transform
        self.resize = resize

        # initialize the queue used to store frames read from
        # the video file
        self.Q = Queue(maxsize=queue_size)
        # intialize thread
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True

    def start(self):
        # start a thread to read frames from the file video stream
        self.thread.start()
        return self

    def update(self):
        # keep looping infinitely
        while True:
            # if the thread indicator variable is set, stop the
            # thread
            if self.stopped:
                break

            # otherwise, ensure the queue has room in it
            if not self.Q.full():
                # read the next frame from the file
                (grabbed, frame) = self.stream.read()

                # if the `grabbed` boolean is `False`, then we have
                # reached the end of the video file
                if not grabbed:
                    self.stopped = True

                # if there are transforms to be done, might as well
                # do them on producer thread before handing back to
                # consumer thread. ie. Usually the producer is so far
                # ahead of consumer that we have time to spare.
                #
                # Python is not parallel but the transform operations
                # are usually OpenCV native so release the GIL.
                #
                # Really just trying to avoid spinning up additional
                # native threads and overheads of additional
                # producer/consumer queues since this one was generally
                # idle grabbing frames.
                if self.transform:
                    frame = self.transform(frame)
                
                if self.resize:
                    frame = cv2.resize(frame, self.resize)

                # add the frame to the queue
                if frame is not None:
                    self.Q.put(frame)
            else:

                time.sleep(0.1)  # Rest for 10ms, we have a full queue

        self.stream.release()

    def read(self):
        # return next frame in the queue
        return self.Q.get()

    def get_batch(self,bs=5):
        bsize = bs if self.Q.qsize() > bs else self.Q.qsize()
        if self.more() or not self.stopped:
            frames = []
            for i in range(bsize):
                frames.append(self.Q.get())
            return frames
        return None

    # Insufficient to have consumer use while(more()) which does
    # not take into account if the producer has reached end of
    # file stream.
    def running(self):
        return self.more() or not self.stopped

    def more(self):
        # return True if there are still frames in the queue. If stream is not stopped, try to wait a moment
        tries = 0
        while self.Q.qsize() == 0 and not self.stopped and tries < 5:
            time.sleep(0.1)
            tries += 1

        return self.Q.qsize() > 0

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True
        # wait until stream resources are released (producer thread might be still grabbing frame)
        self.thread.join()


    @property
    def current_frame_pos(self):
        return self.stream.get(cv2.CAP_PROP_POS_FRAMES)
    
    @property
    def number_of_frames(self):
        return int(self.stream.get(cv2.CAP_PROP_FRAME_COUNT))

    @property
    def frame_rate(self):
        return self.fps
    @property
    def frame_width(self):
        return int(self.fw)

    @property
    def frame_height(self):
        return int(self.fh)

    @property
    def fourcc(self):
        return int(self.stream.get(cv2.CAP_PROP_FOURCC))

    @property
    def frame_format(self):
        return int(self.stream.get(cv2.CAP_PROP_FORMAT))

    # @property
    # def frame_shape(self):
    #     return (self.frame_width, self.frame_height, self.frame_channels)