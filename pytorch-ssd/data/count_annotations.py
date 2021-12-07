import xml.etree.cElementTree as ET
import pathlib
import argparse
import os
import cv2
from collections import Counter

# python data/count_annotations.py --dataset data/helmet
parser = argparse.ArgumentParser(description="Count Class Label Occurrences")
parser.add_argument("--dataset", type=str, help="The root directory of the VOC dataset or Open Images dataset.")
args = parser.parse_args()

if __name__ == '__main__':
    root = pathlib.Path(args.dataset)

    imgnames = []

    # get image names
    for filename in os.listdir(root / "JPEGImages"):
        if filename.endswith(".jpg"):
            img = filename.rstrip('.jpg')
            imgnames.append(img)

    annote_lst = []
    annote = root / "Annotations"

    # Scan the annotations for the labels
    for img in imgnames:
    # img = 'hard_hat_workers2206' # Debug
        img_fname = img + '.xml'
        annote_img = annote / img_fname
        if os.path.isfile(annote_img):
            tree = ET.parse(annote_img)
            root = tree.getroot()
            for labelname in root.findall('*/name'):
                labelname = labelname.text
                # print(img_fname)
                # print(labelname)
                # Produce some sample labels
                if img_fname == 'hard_hat_workers1346.xml':
                    orig_image = cv2.imread("data/helmet/JPEGImages/hard_hat_workers1346.jpg")
                    
                    cv2.rectangle(orig_image, (292,170), (361,290), (255, 255, 0), 4)
                    cv2.putText(orig_image, 'person',
                        (292 - 20, 170 - 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,  # font scale
                        (255, 0, 255),
                        2)  # line type

                    cv2.rectangle(orig_image, (336,171), (361,204), (255, 255, 0), 4)
                    cv2.putText(orig_image, 'helmet',
                        (336 - 10, 171 + 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        .6,  # font scale
                        (255, 0, 255),
                        2)  # line type

                    cv2.imwrite('example_person_helmet_label.jpg', orig_image)
                elif img_fname == 'hard_hat_workers798.xml':
                    orig_image = cv2.imread("data/helmet/JPEGImages/hard_hat_workers798.jpg")
                    
                    cv2.rectangle(orig_image, (207,117), (258,184), (255, 255, 0), 4)
                    cv2.putText(orig_image, 'head',
                        (207 - 20, 117 - 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,  # font scale
                        (255, 0, 255),
                        2)  # line type

                    cv2.imwrite('example_head_label.jpg', orig_image)
                annote_lst.append(labelname)

    print(dict(Counter(annote_lst)))
    print({k: v/len(annote_lst) for k,v in dict(Counter(annote_lst)).items()})
