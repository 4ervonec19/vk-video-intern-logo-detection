from flask import Flask, render_template, request, send_from_directory
import os
from inference_text_detection import *
from inference_text_detection import logo_detection as logo_apply
from inference_embeddings import *
import cv2

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

        print(f"Processed Image 1 Path: {processed_image1_path}")  # Отладочное сообщение
        print(f"Processed Image 2 Path: {processed_image2_path}")  # Отладочное сообщение
        
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
