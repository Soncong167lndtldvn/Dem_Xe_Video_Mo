import cv2
import numpy as np
from ultralytics import solutions

# Kích thước resize video
resize_width = 960
resize_height = 540

# Vùng đếm gốc (phù hợp với video gốc lớn)
original_region = [[0, 300], [1500, 300], [1500, 800], [0, 800]]

# Tự động scale lại vùng đếm cho đúng với khung resize
def scale_region_points(region, orig_w, orig_h, new_w, new_h):
    scale_x = new_w / orig_w
    scale_y = new_h / orig_h
    return [[int(x * scale_x), int(y * scale_y)] for x, y in region]

def enhance_dark_blurry(image):
    #Tăng sáng dùng CLAHE trên kênh V
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    v = clahe.apply(v)
    hsv_enhanced = cv2.merge((h, s, v))
    brightened = cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2BGR)

    #Làm nét ảnh toàn khung
    sharpen_kernel = np.array([[0, -1, 0],
                               [-1, 5, -1],
                               [0, -1, 0]])
    sharpened = cv2.filter2D(brightened, -1, sharpen_kernel)

    #Làm mượt vùng xa phía trên (giảm vỡ hình khi phóng to)
    y1, y2 = 0, int(resize_height * 0.35)  # vùng trên 35% ảnh
    far_area = sharpened[y1:y2, :]

    if far_area.size > 0:
        upscale = cv2.resize(far_area, None, fx=2, fy=2, interpolation=cv2.INTER_LANCZOS4)
        downscale = cv2.resize(upscale, (far_area.shape[1], far_area.shape[0]), interpolation=cv2.INTER_LINEAR)
        sharpened[y1:y2, :] = downscale

    return sharpened

# Hàm chính: đếm vật thể có đi qua vùng đếm
def count_objects_in_region(video_path, output_video_path, model_path):
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), "không thể mở video"

    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    video_writer = cv2.VideoWriter(
        output_video_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (resize_width, resize_height)
    )

    region_points = scale_region_points(original_region, orig_w, orig_h, resize_width, resize_height)

    counter = solutions.ObjectCounter(
        model=model_path,
        region=region_points,
        show=False,
        conf=0.3,
    )

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Resize + xử lý sáng/mờ/mượt vùng xa
        frame = cv2.resize(frame, (resize_width, resize_height))
        enhanced_frame = enhance_dark_blurry(frame)

        # Nhận diện + đếm
        results = counter(enhanced_frame)

        # Ghi video + hiển thị
        video_writer.write(results.plot_im)
        cv2.imshow("kết quả", results.plot_im)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()

# Gọi hàm chính
count_objects_in_region(
    video_path="videomo2.mp4",
    output_video_path="output_scaled.avi",
    model_path="yolov8s.pt"
)
