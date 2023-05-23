import cv2
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('yolov8x-pose-p6.pt')

# Open the video file
video_path = "../data/part1/val/00399/00399.mp4"
cap = cv2.VideoCapture(video_path)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

fps = int(cap.get(cv2.CAP_PROP_FPS))
out = cv2.VideoWriter('./pose.mp4', fourcc, fps, (1280,  720))

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 inference on the frame
        results = model(frame, imgsz=1280, conf=0.5, max_det=2)

        # 5 keypoints for the spine, 4 keypoints for the left arm, 4 keypoints for the right arm, 2 keypoints for the left leg, and 2 keypoints for the right leg.
        keypoint_data = results[0].keypoints.cpu().detach().numpy()
        print(keypoint_data)
        x1l, y1l, x1r, y1r = round(keypoint_data[0][-1][0]), round(keypoint_data[0][-1][1]), round(keypoint_data[0][-2][0]), round(keypoint_data[0][-2][1])
        x2l, y2l, x2r, y2r = round(keypoint_data[1][-1][0]), round(keypoint_data[1][-1][1]), round(keypoint_data[1][-2][0]), round(keypoint_data[1][-2][1])

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        annotated_frame = cv2.circle(annotated_frame, (x1l, y1l), 5, (0,0,0), -1)
        annotated_frame = cv2.circle(annotated_frame, (x1r, y1r), 5, (0,0,0), -1)
        annotated_frame = cv2.circle(annotated_frame, (x2l, y2l), 5, (0,0,0), -1)
        annotated_frame = cv2.circle(annotated_frame, (x2r, y2r), 5, (0,0,0), -1)

        # Display the annotated frame
        cv2.imshow("YOLOv8 Inference", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        # save video
        out.write(annotated_frame)

    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
out.release()
cv2.destroyAllWindows()