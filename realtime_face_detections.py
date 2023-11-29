import os, argparse, cv2
import helper,  numpy as np 
from PIL import Image, ImageDraw, ImageFont
font = ImageFont.truetype("./FiraCode-Regular.ttf", 35)
def builder(args):
    face_detector = helper.get_dlib_face_detector()
    cap = cv2.VideoCapture(args.camera)
    # cap = cv2.VideoCapture("./Addams.mp4")
    colors = ["red","green", "blue", "pink", "orange", "brown"]
    while True:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            o, image = cap.read()
            image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
            landmarks = face_detector(image)
            draw = ImageDraw.Draw(image)
            for i,landmark in enumerate(landmarks):
                linep_s = helper.crop_face(image, landmark, expand=.70)
                color_pic = i%len(colors)
                linep = linep_s[0]
                draw.rectangle([(linep[0],linep[1]),(linep[2],linep[3])], fill=None, outline=colors[color_pic], width=3)
                draw.ellipse([(linep[0],linep[1]),(linep[2],linep[3])], fill=None, outline=colors[color_pic], width=3)
                if(args.crop):
                    face = helper.align_and_crop_face(image, linep_s,160)
                    open_cv_face = np.array(face)
                    # Convert RGB to BGR 
                    open_cv_face = open_cv_face[:, :, ::-1].copy() 
                    cv2.imshow(f"Wajah Ke-{i}",open_cv_face )
                     
                
            open_cv_image = np.array(image)
            # Convert RGB to BGR 
            open_cv_image = open_cv_image[:, :, ::-1].copy() 
            cv2.imshow("Realtime_facedetections", open_cv_image)
            if cv2.waitKey(1) & 0xFF == ord('c'):
                break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--camera', 
        type=int, 
        default=0,
    )
    parser.add_argument(
        '--crop', 
        type=bool, 
        default=False,
    )
    args = parser.parse_args()
    builder(args)
