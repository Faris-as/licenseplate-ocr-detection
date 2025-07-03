import ast
import cv2
import numpy as np
import pandas as pd

def draw_border(img, top_left, bottom_right, color=(0, 255, 0), thickness=10, line_length_x=200, line_length_y=200):
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

license_plate = {}
for car_id in np.unique(results['car_id']):
    max_score = np.amax(results[results['car_id'] == car_id]['license_number_score'])
    best_row = results[(results['car_id'] == car_id) & (results['license_number_score'] == max_score)].iloc[0]

    license_plate[car_id] = {'license_crop': None, 'license_plate_number': best_row['license_number']}

    cap.set(cv2.CAP_PROP_POS_FRAMES, best_row['frame_nmr'])
    ret, frame = cap.read()
    if not ret:
        continue

    try:
        x1, y1, x2, y2 = ast.literal_eval(best_row['license_plate_bbox'].replace('[ ', '[').replace('   ', ' ').replace('  ', ' ').replace(' ', ','))
        license_crop = frame[int(y1):int(y2), int(x1):int(x2), :]
        license_crop = cv2.resize(license_crop, (int((x2 - x1) * 400 / (y2 - y1)), 400))
        license_plate[car_id]['license_crop'] = license_crop
    except Exception as e:
        print(f"[WARNING] Could not crop license plate for car_id {car_id}: {e}")

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
            # Draw green border on car
            car_x1, car_y1, car_x2, car_y2 = ast.literal_eval(df_.iloc[row_indx]['car_bbox'].replace('[ ', '[').replace('   ', ' ').replace('  ', ' ').replace(' ', ','))
            draw_border(frame, (int(car_x1), int(car_y1)), (int(car_x2), int(car_y2)), (0, 255, 0), 25)

            # Draw red license plate box
            x1, y1, x2, y2 = ast.literal_eval(df_.iloc[row_indx]['license_plate_bbox'].replace('[ ', '[').replace('   ', ' ').replace('  ', ' ').replace(' ', ','))
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 12)

            # Overlay license_crop
            license_crop = license_plate[car_id]['license_crop']
            if license_crop is not None:
                H, W, _ = license_crop.shape

                # Ensure boundaries are safe
                if (int(car_y1) - H - 100) > 0 and (int((car_x2 + car_x1 + W) / 2)) < frame.shape[1]:
                    frame[int(car_y1) - H - 100:int(car_y1) - 100,
                          int((car_x2 + car_x1 - W) / 2):int((car_x2 + car_x1 + W) / 2)] = license_crop

                    # White background for text
                    frame[int(car_y1) - H - 400:int(car_y1) - H - 100,
                          int((car_x2 + car_x1 - W) / 2):int((car_x2 + car_x1 + W) / 2)] = (255, 255, 255)

                    (text_width, text_height), _ = cv2.getTextSize(
                        license_plate[car_id]['license_plate_number'],
                        cv2.FONT_HERSHEY_SIMPLEX,
                        4.3,
                        17)

                    cv2.putText(frame,
                                license_plate[car_id]['license_plate_number'],
                                (int((car_x2 + car_x1 - text_width) / 2), int(car_y1 - H - 250 + (text_height / 2))),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                4.3,
                                (0, 0, 0),
                                17)

        except Exception as e:
            print(f"[WARNING] Skipped drawing for car_id {car_id}: {e}")
            continue

    out.write(frame)

# === FINALIZE ===
cap.release()
out.release()
cv2.destroyAllWindows()
print("Visualization complete. Output saved to './out.mp4'")

