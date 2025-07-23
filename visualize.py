import cv2
import numpy as np
import pandas as pd

def parse_bbox(bbox_str):
    if not isinstance(bbox_str, str):
        raise ValueError("bbox_str must be a string")
    cleaned = bbox_str.strip().lstrip('[').rstrip(']')
    if ',' in cleaned:
        parts = cleaned.split(',')
    else:
        parts = cleaned.split()
    return list(map(int, map(float, parts)))  # handles float strings like '123.0'


def draw_border(img, top_left, bottom_right, color=(0, 255, 0), thickness=5, line_length_x=200, line_length_y=200):
    x1, y1 = top_left
    x2, y2 = bottom_right
    cv2.line(img, (x1, y1), (x1, y1 + line_length_y), color, thickness)
    cv2.line(img, (x1, y1), (x1 + line_length_x, y1), color, thickness)
    cv2.line(img, (x1, y2), (x1, y2 - line_length_y), color, thickness)
    cv2.line(img, (x1, y2), (x1 + line_length_x, y2), color, thickness)
    cv2.line(img, (x2, y1), (x2 - line_length_x, y1), color, thickness)
    cv2.line(img, (x2, y1), (x2, y1 + line_length_y), color, thickness)
    cv2.line(img, (x2, y2), (x2, y2 - line_length_y), color, thickness)
    cv2.line(img, (x2, y2), (x2 - line_length_x, y2), color, thickness)
    return img


# === LOAD METADATA ===
results = pd.read_csv('./test_interpolated.csv')

# === LOAD INPUT VIDEO ===
video_path = 'sample.mp4'
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise RuntimeError(f"Failed to open video: {video_path}")

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# === SETUP VIDEO WRITER ===
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('./out.mp4', fourcc, fps, (width, height))

# === Cache license numbers per car_id ===
license_plate = {}
for car_id in np.unique(results['car_id']):
    car_rows = results[results['car_id'] == car_id]
    first_row = car_rows.iloc[0]
    license_plate[car_id] = {'license_crop': None, 'license_plate_number': first_row['license_number']}

# === REWIND VIDEO ===
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
frame_nmr = -1

# === FRAME PROCESSING ===
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_nmr += 1

    df_ = results[results['frame_nmr'] == frame_nmr]

    for row_indx in range(len(df_)):
        car_id = df_.iloc[row_indx]['car_id']

        try:
            # === Draw car box ===
            car_x1, car_y1, car_x2, car_y2 = parse_bbox(df_.iloc[row_indx]['car_bbox'])
            draw_border(frame, (car_x1, car_y1), (car_x2, car_y2), (0, 255, 0), 25)

            # === Get license plate bbox from current frame ===
            x1, y1, x2, y2 = parse_bbox(df_.iloc[row_indx]['license_plate_bbox'])
            x1, x2 = sorted([int(round(x1)), int(round(x2))])
            y1, y2 = sorted([int(round(y1)), int(round(y2))])
            x1 = max(0, min(x1, frame.shape[1] - 1))
            x2 = max(0, min(x2, frame.shape[1]))
            y1 = max(0, min(y1, frame.shape[0] - 1))
            y2 = max(0, min(y2, frame.shape[0]))

            if x2 <= x1 or y2 <= y1:
                continue

            # === Crop license plate from current frame ===
            license_crop = frame[y1:y2, x1:x2, :]
            license_number = license_plate[car_id]['license_plate_number']

            # === Resize license_crop relative to car size ===
            car_width = car_x2 - car_x1
            car_height = car_y2 - car_y1
            plate_target_height = int(car_height * 0.35)
            plate_aspect_ratio = license_crop.shape[1] / license_crop.shape[0]
            plate_target_width = int(plate_aspect_ratio * plate_target_height)

            license_crop_resized = cv2.resize(license_crop, (plate_target_width, plate_target_height))

            # === Calculate overlay position ===
            center_x = (car_x1 + car_x2) // 2
            left_x = max(0, center_x - plate_target_width // 2)
            right_x = min(frame.shape[1], center_x + plate_target_width // 2)
            top_y = max(0, car_y1 - plate_target_height - 20)
            bottom_y = max(0, car_y1 - 20)

            actual_width = right_x - left_x
            if actual_width <= 0:
                continue
            license_crop_resized = cv2.resize(license_crop_resized, (actual_width, plate_target_height))

            # === Draw red license plate bbox ===
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 12)

            # === Draw license plate image ===
            frame[top_y:top_y + plate_target_height, left_x:right_x] = license_crop_resized

            # === Draw white background for text ===
            text_bg_height = int(plate_target_height * 0.8)
            text_bg_top = max(0, top_y - text_bg_height - 10)
            text_bg_bottom = top_y - 10
            frame[text_bg_top:text_bg_bottom, left_x:right_x] = (255, 255, 255)

            # === Draw license number ===
            font_scale = actual_width / 400 * 1.5
            thickness = max(2, int(font_scale * 5))
            (text_width, text_height), _ = cv2.getTextSize(license_number, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            text_x = left_x + (actual_width - text_width) // 2
            text_y = text_bg_top + (text_bg_height + text_height) // 2

            cv2.putText(frame, license_number, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)

        except Exception as e:
            print(f"[WARNING] Skipped drawing for car_id {car_id}: {e}")
            continue

    out.write(frame)

# === FINALIZE ===
cap.release()
out.release()
cv2.destroyAllWindows()
print("✅ Visualization complete. Output saved to './out.mp4'")

