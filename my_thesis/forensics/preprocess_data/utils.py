import cv2
import numpy as np
import sys
import os.path as osp
print(sys.path)

def vis_face_img(image: np.ndarray, pos:np.ndarray, prob:np.ndarray, landmark=None, vis=True, save=True):
    """
        Draw and save faces
    """
    img = cv2.cvtColor(image, code=cv2.COLOR_RGB2BGR).copy()
    pos = np.array(pos, dtype=np.int32).reshape(2, 2)
    img = cv2.rectangle(img, pos[0], pos[1], (255, 0, 0), thickness=1)
    img = cv2.putText(img, str(prob), pos[0], cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), thickness=1)
    if landmark is not None:
        e = 5
        landmark = np.array(landmark, dtype=np.int32)
        for p in landmark:
            img = cv2.rectangle(img, (p[0]-e, p[1]-e), (p[0]+e, p[1]+e), (0, 0, 255))
    if vis:
        cv2.imshow("Face image", img)
        cv2.waitKey(0)
    if save:
        cv2.imwrite(osp.join(osp.dirname(__file__), "test/face_extract.png"), img)