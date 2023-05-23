# [AI CUP 2023 - Teaching Computer to Watch Badminton Matches - Taiwan's first competition combining AI and sports](https://aidea-web.tw/topic/cbea66cc-a993-4be8-933d-1aa9779001f8)

## TEAM_3379: Celia

## Introduction
Global statistics indicate that there are approximately 2.2 billion badminton players worldwide, with over 3 million in Taiwan alone. Badminton ranks second in terms of national popularity. Recently, badminton players have been achieving remarkable performances in international competitions, attracting increased public attention.

To analyze badminton skills and tactics, our team has introduced a match shuttlecock recording format and developed a computer vision-assisted program for quick shuttlecock labeling. While various computer-assisted techniques have been employed, manual shuttlecock labeling still requires time and manpower, especially for technical data identification, which necessitates the expertise of badminton professionals. With this competition, we aim to engage machine learning, image processing, and sports science specialists in developing an automatic shuttlecock labeling model with a high recognition rate. This would enable the collection of extensive badminton information and promote the research and application of badminton tactics analysis.

## Grades

<table>
  <tr>
    <td>Public Grade</td>
    <td>Public Ranking</td>
    <td>Private Grade</td>
    <td>Private Ranking</td>
  </tr>
  <tr>
    <td>0.0548</td>
    <td>16</td>
    <td>0.0324</td>
    <td>17</td>
  </tr>
</table>

## Datasets

- [part1.zip](https://drive.google.com/file/d/1h5qRYnE2scuMGIJUq2SRWW2KLol6wMyh/view?usp=share_link)
- [part2.zip](https://drive.google.com/file/d/1SLY5YM4Q61N6DmqPuSUNzUANQ0s4mjX5/view?usp=share_link)

## Folder Structure

    .
    ├── data                            # contains val and test videos
    │   ├── val                         # predict file after execute test.py
    │       └── 00001, 00002, ...       # val and test folders
    │   └── Hit                         # hitframe extracted from test.csv
    ├── The_Demo_529                    # get the trajectory of badminton
    │   ├── test.py                     # exectuted file
    │   ├── rand.py                     # exectuted file
    │   ├── predict                     # predict files after execute test.py
    │       └── 00001.csv, 00001.jpg, 00001.mp4, 00001_predict_shot.csv, ...
    │   ├── test.csv                    # generated from test.py
    │   └── test_rand.csv               # generated from rand.py
    └── ultralytics                     # get the pose information
        ├── pose.py                     # exectuted file
        └── submit.csv                  # generated from rand.py


## Reproduce the Result

### Install Libraries

    $ conda create -n badminton python=3.7
    $ conda activate badminton
    $ git clone https://nol.cs.nctu.edu.tw:234/lukelin/The_Demo_529.git
    $ sudo apt-get install git
    $ sudo apt-get install python3-pip
    $ pip3 install pyqt5
    $ pip3 install pandas
    $ pip3 install PyMySQL
    $ pip3 install opencv-python
    $ pip3 install matplotlib
    $ pip3 install pytorch
    $ pip3 install imutils
    $ pip3 install Pillow
    $ pip3 install piexif
    $ pip3 install -U scikit-learn
    $ pip3 install keras
    $ git clone https://github.com/ultralytics/ultralytics.git
    $ cd ultralytics/
    $ pip3 install -r requirements.txt

### Execution Steps

    $ cd The_Demo_529/
    $ python test.py
    $ python -W ignore rand.py
    $ cd ultralytics/
    $ wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x-pose-p6.pt
    $ python pose.py

## Demo

https://github.com/celia9217/AICUP-Badminton-Competition/assets/63925539/77948e96-d898-41e9-9542-779699f8bd31

https://github.com/celia9217/AICUP-Badminton-Competition/assets/63925539/0477b7ad-19a2-4883-96a6-42fcd83ef158



## Reference

- [The combination of TrackNetV2, Trajectory Smoothing, Event Detection, YoloV3 and Pseudo3D](https://nol.cs.nctu.edu.tw:234/lukelin/The_Demo_529)
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
  - [Pose](https://docs.ultralytics.com/tasks/pose/)
  - [Working with Results](https://docs.ultralytics.com/modes/predict/#working-with-results)
  - [Streaming Source for-loop](https://docs.ultralytics.com/modes/predict/#streaming-source-for-loop)
  - [YOLOv8 pose-estimation model](https://github.com/ultralytics/ultralytics/issues/2028)
- [How to disable Python warnings?](https://stackoverflow.com/questions/14463277/how-to-disable-python-warnings)
