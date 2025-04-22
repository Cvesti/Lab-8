import cv2
import numpy as np

# Функция для сохранения координат в файл
def save_coordinates_to_file(x, y):
    with open('marker_coordinates.txt', 'a') as f:
        f.write(f"Marker coordinates: x={x}, y={y}\n")

# Загрузка изображения метки
marker = cv2.imread('images/variant-2.png', cv2.IMREAD_GRAYSCALE)
if marker is None:
    print("Error: Could not load marker image")
    exit()

# Инициализация детектора ORB
orb = cv2.ORB_create()
kp_marker, desc_marker = orb.detectAndCompute(marker, None)

# Инициализация видеозахвата
cap = cv2.VideoCapture(0)

# Создание BFMatcher объекта
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Создаем окна с возможностью изменения размера
cv2.namedWindow('Matches', cv2.WINDOW_NORMAL)
cv2.namedWindow('Blurred Frame', cv2.WINDOW_NORMAL)

# Устанавливаем начальный размер окон
cv2.resizeWindow('Matches', 800, 600)
cv2.resizeWindow('Blurred Frame', 800, 600)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Конвертация в градации серого
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Применение размытия по Гауссу
    blurred_frame = cv2.GaussianBlur(gray_frame, (7, 7), 0)
    
    # Поиск ключевых точек на кадре
    kp_frame, desc_frame = orb.detectAndCompute(blurred_frame, None)
    
    if desc_frame is not None and len(kp_frame) > 0:
        # Сопоставление дескрипторов
        matches = bf.match(desc_marker, desc_frame)
        matches = sorted(matches, key=lambda x: x.distance)
        
        # Отображение лучших совпадений
        good_matches = matches[:10]
        img_matches = cv2.drawMatches(marker, kp_marker, blurred_frame, kp_frame, 
                                     good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        
        # Получение координат центра метки
        if len(good_matches) > 0:
            # Используем среднее положение всех хороших совпадений
            sum_x = 0
            sum_y = 0
            count = 0
            
            for match in good_matches:
                frame_idx = match.trainIdx
                x, y = kp_frame[frame_idx].pt
                sum_x += x
                sum_y += y
                count += 1
            
            if count > 0:
                center_x = int(sum_x / count)
                center_y = int(sum_y / count)
                
                # Сохранение координат в файл
                save_coordinates_to_file(center_x, center_y)
                
                # Отрисовка центра метки
                cv2.circle(blurred_frame, (center_x, center_y), 10, (0, 255, 0), 2)
                cv2.putText(blurred_frame, f"({center_x}, {center_y})", 
                            (center_x + 15, center_y), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.5, (0, 255, 0), 2)
        
        cv2.imshow('Matches', img_matches)
    
    cv2.imshow('Blurred Frame', blurred_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()