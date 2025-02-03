from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator

from inference_sdk import InferenceHTTPClient
import pytesseract
import easyocr

from torchvision.transforms import transforms

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import Levenshtein


import argparse
import os
import warnings
warnings.filterwarnings('ignore')

current_dir = os.path.dirname(os.path.abspath(__file__))

yolo_weights_path = os.path.join(current_dir, '..', 'notebooks', 'yolo', 'yolo_weigths', 'best.pt')
API_KEY = os.getenv('API_KEY')

# Базовый трансформ данных
base_transform = transforms.Compose(
    [
        transforms.Resize((300, 300)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

def logo_detection(image_path:str, 
                yolo_weights_path:str = yolo_weights_path, 
                conf_threshold:int = 0.15, 
                plot_results:bool = False):
    '''
    Функция получения результатов детекции логотипов
    
    Параметры:
    -yolo_weights_path: путь к сохранённым весами модели YOLO (str, default: yolo_weights_path)
    -conf_threshold: порог уверенности детекции (int, default: 0.25)
    -plot_results: флаг для построения изображений (bool, default: False)
    '''
    
    yolo_logo_detection = YOLO(yolo_weights_path)
    result = yolo_logo_detection(image_path, verbose = False)[0]
    initial_image = Image.open(image_path).copy()

    if plot_results:
        plt.imshow(initial_image)
        plt.show()

    initial_image = np.asarray(Image.open(image_path))
    
    annotator = Annotator(result.orig_img)
    boxes = result.boxes
    for box in boxes:
        b = box.xyxy[0]
        c = box.cls
        conf = box.conf
        if conf > conf_threshold:
            annotator.box_label(b, yolo_logo_detection.names[int(c)])
    
    img = annotator.result() 

    if plot_results:
        print(f"Inference result:")
        plt.imshow(img)
        plt.show()
    
    cropped_images = []
    if plot_results:
        print(f"Cropped results:")
    for i, box in enumerate(boxes):
        b = box.xyxy[0].cpu().numpy()
        x_min, y_min, x_max, y_max = b[0], b[1], b[2], b[3]
        x_min, y_min, x_max, y_max = map(int, [x_min, y_min, x_max, y_max])
        cropped_image = initial_image[y_min:y_max, x_min:x_max]
        cropped_images.append(cropped_image)
        if plot_results:
            plt.imshow(cropped_image)
            plt.show()
    
    cropped_images.append(initial_image)

    return cropped_images, img

# Step 1. cropped_images = logo_detection(plot_results=True, image_path = ...)
# cropped_images_logo = logo_detection(plot_results=True, image_path = ...)

def get_text_predictions(cropped_images, api_key):
    '''
    Функция предсказания текста по cropped результатам Object Detection

    Параметры:
    -cropped_images: массив полученных обрезанных изображений (default: cropped_images)
    -api_key: API_KEY roboflow.com
    '''

    CLIENT = InferenceHTTPClient(
        api_url="https://infer.roboflow.com",
        api_key=api_key
    )

    text_results = []
    reader = easyocr.Reader(['en'])

    for image in cropped_images:
        # result_roboflow = CLIENT.ocr_image(inference_input=image) # Опционально, почему-то локально работает через API-KEY
                                                                    # Через docker не работает
        result_tesseract = pytesseract.image_to_string(image)
        result_easyocr = reader.readtext(image)
        # text_results.append(result_roboflow['result'])
        text_results.append(result_tesseract)

        for (bbox, text, prob) in result_easyocr:
            text_results.append(text)
        break

    return text_results

# Step 2. text_results = get_text_predictions(cropped_images=cropped_images,
# 									api_key=API_KEY)
# text_results_logo = get_text_predictions(cropped_images=cropped_images_logo,
# 									api_key=API_KEY)

def levenstein_metric(cropped_images_predicted, 
            cropped_images_logo):
    '''
    Функция подсчёта метрки Левенштейна
    
    Параметры:
    -cropped_images_predicted: полученные тексты из кадра видео
    -cropped_images_logo: полученные тексты из логотипа
    
    Возвращает:
    -метрика левенштейна (немного изменённая для интерпретации от 0 до 1)
    '''
    
    max_metric = 0
    for pred in cropped_images_predicted:
        for logo in cropped_images_logo:
            pred_filtered = ''.join([char for char in pred if char.isalpha()])
            logo_filtered = ''.join([char for char in logo if char.isalpha()])
            distance = Levenshtein.distance(logo_filtered, pred_filtered)
            max_len = max(len(pred_filtered), len(logo_filtered)) 
            normalized_distance = distance / max_len if max_len != 0 else 1
            if 1 - normalized_distance > max_metric and len(pred_filtered) != 0 and len(logo_filtered) != 0:
                max_metric = 1 - normalized_distance

    return max_metric

# Step 3. levenstein_distance = levenstein_metric(text_results_logo, text_results)

def get_text_metric(logo_searched:str, 
                    video_frame:str, 
                    yolo_path:str = yolo_weights_path,
                    api_key:str = API_KEY):
    
    '''
    Функция для подсчета метрики близости между текстовыми представлениями, извлеченными из изображений

    Параметры:
    -logo_searched: путь к логотипу
    -video_frame: путь к кадру видео
    -yolo_path: путь к YOLO
    -api_key: ключ API на roboflow.com
    '''
    
    cropped_images, _ = logo_detection(plot_results=False, yolo_weights_path=yolo_path,
                                    image_path = video_frame)
    
    cropped_images_logo, _ = logo_detection(plot_results=False, yolo_weights_path=yolo_path,
                                    image_path = logo_searched)
    
    video_frame_text_pred = get_text_predictions(cropped_images=cropped_images,
                                    api_key=api_key)
    
    logo_text_pred = get_text_predictions(cropped_images=cropped_images_logo,
                                    api_key=api_key)
    
    lev_metric_res = levenstein_metric(video_frame_text_pred, logo_text_pred)

    return lev_metric_res

# Step 4. lev_metric_calc = get_text_metric(logo_searched=logo_path,
#                         video_frame=image_path_, 
#                         yolo_path=yolo_weights_path,
#                         api_key=API_KEY)

def main(logo_path, image_path):
    lev_metric_calc = get_text_metric(
        logo_searched=logo_path,
        video_frame=image_path,
        yolo_path=yolo_weights_path,
        api_key=API_KEY
    )
    print(f'Levenstein metric: {lev_metric_calc}')

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description='Levenstein metric.')
#     parser.add_argument('logo_path', type=str, help='First image path.')
#     parser.add_argument('image_path', type=str, help='Second image path.')

#     args = parser.parse_args()

#     print(args)

#     if not os.path.exists(args.logo_path):
#         print(f'File {args.logo_path} not found.')
#         exit(1)

#     if not os.path.exists(args.image_path_):
#         print(f'File {args.image_path} not found.')
#         exit(1)

#     main(args.logo_path, args.image_path)












