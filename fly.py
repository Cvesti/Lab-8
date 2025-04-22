import cv2
import numpy as np

def save_coordinates_to_file(x, y):
    with open('marker_coordinates.txt', 'a') as f:
        f.write(f"Marker coordinates: x={x}, y={y}\n")

# Загрузка изображения метки и мухи
marker = cv2.imread('images/variant-2.png', cv2.IMREAD_GRAYSCALE)
fly = cv2.imread('fly64.png', cv2.IMREAD_UNCHANGED)  # Загружаем с альфа-каналом

if marker is None or fly is None:
    print("Error: Could not load images")
    exit()

# Инициализация ORB
orb = cv2.ORB_create()
kp_marker, desc_marker = orb.detectAndCompute(marker, None)

# Подготовка мухи (разделяем цвет и альфа-канал)
fly_bgr = fly[:, :, :3]
fly_alpha = fly[:, :, 3] / 255.0  # Нормализуем альфа-канал

cap = cv2.VideoCapture(0)

# Создаем окна с возможностью изменения размера
cv2.namedWindow('Tracking', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Tracking', 800, 600)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)
    
    kp_frame, desc_frame = orb.detectAndCompute(blurred_frame, None)
    
    if desc_frame is not None and len(kp_frame) > 0:
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(desc_marker, desc_frame)
        matches = sorted(matches, key=lambda x: x.distance)
        
        good_matches = matches[:10]
        
        if len(good_matches) > 0:
            sum_x = sum_y = count = 0
            for match in good_matches:
                x, y = kp_frame[match.trainIdx].pt
                sum_x += x
                sum_y += y
                count += 1
            
            center_x = int(sum_x / count)
            center_y = int(sum_y / count)
            
            save_coordinates_to_file(center_x, center_y)
            
            # Наложение изображения мухи
            fly_height, fly_width = fly.shape[:2]
            
            # Вычисляем область для вставки мухи
            y1 = max(0, center_y - fly_height//2)
            y2 = min(frame.shape[0], center_y + fly_height//2)
            x1 = max(0, center_x - fly_width//2)
            x2 = min(frame.shape[1], center_x + fly_width//2)
            
            # Если муха выходит за границы кадра, корректируем размеры
            if x2 - x1 > 0 and y2 - y1 > 0:
                # Область в кадре, куда будем вставлять муху
                roi = frame[y1:y2, x1:x2]
                
                # Соответствующая часть мухи
                fly_part = fly_bgr[:y2-y1, :x2-x1]
                alpha_part = fly_alpha[:y2-y1, :x2-x1]
                
                # Наложение с учетом прозрачности
                for c in range(3):
                    roi[:, :, c] = roi[:, :, c] * (1 - alpha_part) + fly_part[:, :, c] * alpha_part
                
                # Отметка центра (для отладки)
                cv2.circle(frame, (center_x, center_y), 3, (0, 0, 255), -1)
    
    cv2.imshow('Tracking', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()