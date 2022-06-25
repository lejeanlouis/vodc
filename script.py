import numpy as np
import cv2 as cv
import sys
import json

def main():
    while True:
        command = sys.stdin.readline()
        obbie_in = json.loads(command)
        frame_id = int(obbie_in['frame_id']) + 1

        x = int(obbie_in['x'])
        y = int(obbie_in['y'])
        w = int(obbie_in['w'])
        h = int(obbie_in['h'])

        x = int(x - (w / 10))
        y = int(y)
        w = int(w / 5)
        h = int(h / 10)

        try:
           image = cv.imread(
               "../darknet/saved_frames/image_%08d.jpg" % frame_id)
           crop = image[y:y+h, x:x+w]
           pixels = np.float32(crop.reshape(-1, 3))
           n_colors = 10
           criteria = (cv.TERM_CRITERIA_EPS + cv.KMEANS_PP_CENTERS, 200, .1)
           flags = cv.KMEANS_RANDOM_CENTERS
           _, labels, palette = cv.kmeans(
               pixels, n_colors, None, criteria, 10, flags)
           _, counts = np.unique(labels, return_counts=True)
           dominant = palette[np.argmax(counts)]
        except:
            try:
                image = cv.imread(
                    "../darknet/saved_frames/image_%08d.jpg" % frame_id)
                crop = image[y:y+h, x:x+w]
                pixels = np.float32(crop.reshape(-1, 3))
                n_colors = 10
                criteria = (cv.TERM_CRITERIA_EPS +
                            cv.KMEANS_PP_CENTERS, 200, .5)
                flags = cv.KMEANS_RANDOM_CENTERS
                _, labels, palette = cv.kmeans(
                    pixels, n_colors, None, criteria, 10, flags)
                _, counts = np.unique(labels, return_counts=True)
                dominant = palette[np.argmax(counts)]
            except:
                continue

        obbie_out = {
            'recordingId': obbie_in['recordingId'],
            'frame_id': obbie_in['frame_id'],
            'key': obbie_in['key'],
            'color': '#%02x%02x%02x' % (int(dominant[2]), int(dominant[1]), int(dominant[0])),
            'log': '#%02x%02x%02x' % (int(dominant[2]), int(dominant[1]), int(dominant[0])) + ' --x%d y%d w%d g%d f%d' % (x, y, w, h, frame_id)
        }
        sys.stdout.write(json.dumps(obbie_out) + "\n")
        sys.stdout.flush()


if __name__ == '__main__':
    main()
