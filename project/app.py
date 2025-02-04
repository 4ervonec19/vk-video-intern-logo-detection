from flask import Flask, render_template, request, send_from_directory
import os
from inference_text_detection import *
from inference_text_detection import logo_detection as logo_apply
from inference_embeddings import *
import cv2
import sqlite3

app = Flask(__name__)
UPLOAD_FOLDER_LOGO = 'logos'
UPLOAD_FOLDER_FRAME = 'frames'
PROCESSED_FOLDER_LOGO = 'processed_logos'
PROCESSED_FOLDER_FRAME = 'processed_frames'
os.makedirs(UPLOAD_FOLDER_LOGO, exist_ok=True)
os.makedirs(UPLOAD_FOLDER_FRAME, exist_ok=True)
os.makedirs(PROCESSED_FOLDER_LOGO, exist_ok=True)
os.makedirs(PROCESSED_FOLDER_FRAME, exist_ok=True)
app.config['UPLOAD_FOLDER_LOGO'] = UPLOAD_FOLDER_LOGO
app.config['UPLOAD_FOLDER_FRAME'] = UPLOAD_FOLDER_FRAME
app.config['PROCESSED_FOLDER_LOGO'] = PROCESSED_FOLDER_LOGO
app.config['PROCESSED_FOLDER_FRAME'] = PROCESSED_FOLDER_FRAME

counter = 0

def get_connection():
    '''Функция создания и подлкючения к БД для логирования и записи результатов'''
    
    connection = sqlite3.connect('logs.db')
    cursor = connection.cursor()
    CREATE_LOGO_TABLE_QUERY = '''CREATE TABLE IF NOT EXISTS logo_images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image BLOB NOT NULL
        )'''
    
    CREATE_FRAME_TABLE_QUERY = '''CREATE TABLE IF NOT EXISTS frame_images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image BLOB NOT NULL
        )'''
    
    CREATE_PAIRS_TABLE_QUERY = '''CREATE TABLE IF NOT EXISTS images_processing (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            id_logo INTEGER,
            id_frame INTEGER,
            levenstein_metric FLOAT,
            cosine_similarity FLOAT,
            probability FLOAT,
            FOREIGN KEY (id_logo) REFERENCES logo_images(id),
            FOREIGN KEY (id_frame) REFERENCES frame_images(id)
        )'''
    
    cursor.execute(CREATE_LOGO_TABLE_QUERY)
    cursor.execute(CREATE_FRAME_TABLE_QUERY)
    cursor.execute(CREATE_PAIRS_TABLE_QUERY)

    connection.commit()
    return connection

def save_image_to_db(cursor, image_path, table_name):
    '''Сохранение изображения в таблицу'''
    with open(image_path, 'rb') as file:
        blob_data = file.read()
    cursor.execute(f'INSERT INTO {table_name} (image) VALUES (?)', (blob_data,))
    return cursor.lastrowid

def save_processing_results(cursor, id_logo, id_frame, levenstein_metric, cosine_similarity, probability):
    '''Вставка полученных данных в БД'''
    cursor.execute('''
        INSERT INTO images_processing (id_logo, id_frame, levenstein_metric, cosine_similarity, probability)
        VALUES (?, ?, ?, ?, ?)
    ''', (id_logo, id_frame, levenstein_metric, cosine_similarity, probability))
    
@app.route('/', methods=['GET', 'POST'])
def upload_files():
    global counter
    if request.method == 'POST':
        if 'image1' not in request.files or 'image2' not in request.files:
            return "Error: Upload 2 Images"
        
        image1 = request.files['image1']
        image2 = request.files['image2']
        
        if image1.filename == '' or image2.filename == '':
            return "Error: Choose Files"
        
        image1_path = os.path.join(app.config['UPLOAD_FOLDER_LOGO'], 'initial_' + str(counter) + '_logo.jpg')
        image2_path = os.path.join(app.config['UPLOAD_FOLDER_FRAME'], 'initial_' + str(counter) + '_frame.jpg')
        
        image1.save(image1_path)
        image2.save(image2_path)
        
        processed_image1_path = os.path.join(app.config['PROCESSED_FOLDER_LOGO'], 'processed_' + str(counter) + '_logo.jpg')
        processed_image2_path = os.path.join(app.config['PROCESSED_FOLDER_FRAME'], 'processed_' + str(counter) + '_frame.jpg')

        counter += 1

        # Image preprocessing

        _, img1 = logo_apply(image_path=image1_path)
        _, img2 = logo_apply(image_path=image2_path)

        cv2.imwrite(processed_image1_path, img1)
        cv2.imwrite(processed_image2_path, img2)

        levenstein_distance = get_text_metric(logo_searched=image2_path, 
                                              video_frame=image1_path)
        
        cosine_similarity = get_cosine_similarity(logo_searched=image2_path, 
                                              video_frame=image1_path).item()
        
        mean_result = (levenstein_distance + cosine_similarity) / 2
        
        # print(f"Levenstein Metric: {levenstein_distance:.2f}")
        # print(f"Cosine Similarity Metric: {cosine_similarity:.2f}")
        # print(f"Mean value: {mean_result:.2f}")

        # Подключение к БД
        connection = get_connection()
        cursor = connection.cursor()

        # Сохранение изображениия и получение id
        id_logo = save_image_to_db(cursor, image1_path, 'logo_images')
        id_frame = save_image_to_db(cursor, image2_path, 'frame_images')

        # Сохранение результатов обработки
        save_processing_results(cursor, id_logo, id_frame, levenstein_distance, cosine_similarity, mean_result)

        connection.commit()
        connection.close()

        # print(f"Processed Image 1 Path: {processed_image1_path}")  # Отладочное сообщение
        # print(f"Processed Image 2 Path: {processed_image2_path}")  # Отладочное сообщение
        
        return render_template('upload.html', 
                               image1=processed_image1_path, 
                               image2=processed_image2_path,
                               levenstein=levenstein_distance,
                               cosine=cosine_similarity,
                               mean=mean_result)
    
    return render_template('upload.html', image1=None, image2=None)

@app.route('/processed/<filename>')
def processed_file(filename):
    if 'logo' in filename:
        folder = app.config['PROCESSED_FOLDER_LOGO']
    else:
        folder = app.config['PROCESSED_FOLDER_FRAME']

    file_path = os.path.join(folder, filename)
    if not os.path.exists(file_path):
        return "File not found", 404
    
    return send_from_directory(folder, filename)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8002)
