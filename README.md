# Badminton-Competition 

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

## Folder Structure
    ```
    .
    ├── data                            # contains val and test videos
    │   └── val                         # predict file after execute test.py
    │       └── 00001, 00002, ...       # val and test folders
    ├── The_Demo_529                    # get the trajectory of badminton
    │   ├── test.py                     # exectuted file
    │   └── predict                     # predict files after execute test.py
    │       └── 00001.csv, 00001.jpg, 00001.mp4, 00001_predict_shot.csv, ...
    └── ultralytics                     # get the pose information
    ```

## Reproduce the Result

### Install Libraries
    ```
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
    ```

### Execution Steps
    ```
    $ cd The_Demo_529/
    $ python test.py
    $ cd ultralytics/
    ```

## Reference

- [The combination of TrackNetV2, Trajectory Smoothing, Event Detection, YoloV3 and Pseudo3D](https://nol.cs.nctu.edu.tw:234/lukelin/The_Demo_529)
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
