import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
from tracker import Tracker


def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        colorsBGR = [x, y]
        print(colorsBGR)


def main():
    model = YOLO('yolov8s.pt')

    area1 = [(282, 378), (259, 380), (504, 479), (527, 472)]
    area2 = [(249, 382), (220, 387), (453, 487), (484, 479)]

    cv2.namedWindow('RGB')
    cv2.setMouseCallback('RGB', RGB)

    cap = cv2.VideoCapture('data/input_video.mp4')

    my_file = open("coco.txt", "r")
    data = my_file.read()
    class_list = data.split("\n")
    # print(class_list)


    people_ent = {}
    people_exit = {}
    entering = set()
    exiting = set()
    count = 0
    tracker = Tracker()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        count += 1
        if count % 2 != 0:
            continue
        frame = cv2.resize(frame, (1020, 500))
        #    frame=cv2.flip(frame,1)
        results = model.predict(frame)
        #   print(results)
        a = results[0].boxes.data
        a_np = [t.cpu().numpy() for t in a]
        px = pd.DataFrame(a_np).astype("float")

        coords = []
        for index, row in px.iterrows():

            x1, y1, x2, y2 = int(row[0]), int(row[1]), int(row[2]), int(row[3])
            label = int(row[5])
            class_name = class_list[label]
            if 'person' in class_name:
                coords.append([x1,y1,x2,y2])

        bbox_id= tracker.update(coords)
        for bbox in bbox_id:
            x3, y3, x4, y4, id = bbox
            results_for_area2 = cv2.pointPolygonTest(np.array(area2, np.int32), ((x4,y4)), False)
            if results_for_area2>=0:
                people_ent[id]=(x4,y4)
                cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 0), 2)
                cv2.circle(frame,(x4,y4), 6, (0,0,255), -1)
                cv2.putText(frame, str(id), (x4, y4), cv2.FONT_HERSHEY_COMPLEX, (0.5), (255, 255, 255), 1)
                entering.add(id)

                if id in people_ent:
                    results_for_area1 = cv2.pointPolygonTest(np.array(area1, np.int32), ((x4, y4)), False)
                    if results_for_area1 >= 0:
                        cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 0), 2)
                        cv2.circle(frame, (x4, y4), 6, (0, 0, 0), -1)
                        cv2.putText(frame, str(id), (x3, y3), cv2.FONT_HERSHEY_COMPLEX, (0.5), (255, 255, 255), 1)
                        entering.add(id)

            results2_for_area1 = cv2.pointPolygonTest(np.array(area1, np.int32), ((x4, y4)), False)
            if results2_for_area1 >= 0:
                people_exit[id] = (x4, y4)
                cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 0), 2)
                cv2.circle(frame, (x4, y4), 6, (0,0,255), -1)
                cv2.putText(frame, str(id), (x3, y3), cv2.FONT_HERSHEY_COMPLEX, (0.5), (255, 255, 255), 1)
                entering.add(id)

            if id in people_exit:
                results2_for_area2 = cv2.pointPolygonTest(np.array(area2, np.int32), ((x4, y4)), False)
                if results2_for_area2 >= 0:
                    cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 0), 2)
                    cv2.circle(frame, (x4, y4), 6, (0,0,255), -1)
                    cv2.putText(frame, str(id), (x3, y3), cv2.FONT_HERSHEY_COMPLEX, (0.5), (255, 255, 255), 1)
                    exiting.add(id)



        cv2.polylines(frame, [np.array(area1, np.int32)], True, (255, 0, 0), 2)
        cv2.putText(frame, str('1'), (504, 471), cv2.FONT_HERSHEY_COMPLEX, (0.5), (0, 0, 0), 1)

        cv2.polylines(frame, [np.array(area2, np.int32)], True, (0, 255, 255), 2)
        cv2.putText(frame, str('2'), (466, 485), cv2.FONT_HERSHEY_COMPLEX, (0.5), (0, 0, 0), 1)
        print(entering)
        print(len(entering))
        print(exiting)
        print(len(exiting))
        total = len(entering) - len(exiting)
        print(total)
        cv2.putText(frame, str(total), (60, 80), cv2.FONT_HERSHEY_COMPLEX, (2), (255, 255, 255), 4)


        cv2.imshow("RGB", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break




    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
