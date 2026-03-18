import os
os.environ['APP_DATA_DIR'] = '/home1/lu-wei/repo/celeb-detection-oss/examples/resources/'

import argparse
import moviepy.editor as mov_editor
from dotenv import load_dotenv
from skimage import io
from pprint import pprint
from model_training.utils import preprocess_image
from model_training.helpers.labels import Labels
from model_training.helpers.face_recognizer import FaceRecognizer
from model_training.utils import evenly_spaced_sampling
from model_training.preprocessors.face_detection.face_detector import FaceDetector
from tqdm import tqdm
import pandas as pd
import re


def process_image(path):
    image = io.imread(path)
    face_images = face_detector.perform_single(image)
    face_images = [preprocess_image(image, image_size) for image, _ in face_images]
    return face_recognizer.perform(face_images)


def extract_celebrity_name(text):

    # evaluation patterns
    patterns = [
        r"A portrait of (.*)_(\d+)\.png",
        r"A photo of (.*)_(\d+)\.png",
        r"An image capturing (.*) at a public event_(\d+)\.png",
        r"An oil painting of (.*)_(\d+)\.png",
        r"A sketch of (.*)_(\d+)\.png",
        r"(.*) in an official photo_(\d+)\.png",
        r"a_photo_of_(.*)_(\d+)\.jpg", 
    ]
    no_match = True

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)

        # match = re.search(pattern, text)
        if match:  
            return match.group(1)  
        
    if no_match:
        print(text)
        raise ValueError("The input image name does not match any of the expected patterns.")

def format_celebrity_name(name):
    return name.replace('-', ' ').title()


if __name__ == '__main__':
    load_dotenv('.env')
    parser = argparse.ArgumentParser(description='Inference script for Giphy Celebrity Classifier model')
    parser.add_argument('--image_folder', type=str, help='path or link to the image folder', default=None)
    parser.add_argument('--save_excel_path', type=str, help='path to save the excel file', default=None)
    parser.add_argument('--save_csv_path', type=str, help='path to save the csv file', default=None)
    parser.add_argument('--cele_name', type=str, help='erased celebrity name', default=None)
    parser.add_argument('--evaluation_metric', type=str, help='evaluation metrics for erasure', default=None)

    args = parser.parse_args()

    image_size = int(os.getenv('APP_FACE_SIZE', 224))
    gif_frames = int(os.getenv('APP_GIF_FRAMES', 20))

    model_labels = Labels(resources_path=os.getenv('APP_DATA_DIR'))

    face_detector = FaceDetector(
        os.getenv('APP_DATA_DIR'),
        margin=float(os.getenv('APP_FACE_MARGIN', 0.2)),
        use_cuda=os.getenv('APP_USE_CUDA') == "true"
    )
    face_recognizer = FaceRecognizer(
        labels=model_labels,
        resources_path=os.getenv('APP_DATA_DIR'),
        use_cuda=os.getenv('USE_CUDA') == "true",
        top_n=5 
    )

    # image_files=os.listdir(args.image_folder)
    # image_names=sorted(image_files)   #sort image files
    
    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    image_names = sorted([f for f in os.listdir(args.image_folder) if f.lower().endswith(valid_exts)])
    
    predictions_list=[]
    p_celebrity_list=[]  
    n_no_faces=0
    valid_image_names = []

    
    for file in tqdm(image_names):
        image_path=os.path.join(args.image_folder,file)
        
        try:
            predictions = process_image(image_path)
        except ValueError as e:
            print(f"Skipping {image_path} due to error: {e}")
            continue  # 或者 pass，视你是否在循环中而定
        
        valid_image_names.append(file) 
        # predictions = process_image(image_path) # precdictions contain the probabilities of the top n celebrities for one image
        if len(predictions)==0:     # if no face detected
            n_no_faces+=1
            p_celebrity_list.append('N')  # give empty string if no face detected
            predictions_list.append([])
        else:
            predictions_new_label=[]
            for prediction in predictions[0][0]:
                celebrity_label, prob=prediction
                celebrity_label=str(celebrity_label)  
                # Modify label format
                celebrity_name=celebrity_label.split('_[',1)[0].replace('_',' ')
                prediction=(celebrity_name,prob)
                predictions_new_label.append(prediction)
            predictions_list.append(predictions_new_label)

            print('************************')
            if args.evaluation_metric in ["efficacy_01", "efficacy_02", "efficacy_03", "efficacy_04"]:
                ground_truth_name = format_celebrity_name(args.cele_name)
            else:
                ground_truth_name = extract_celebrity_name(file)

            print("Predicted:", predictions_new_label[0][0])
            print("Ground Truth:", ground_truth_name)
            if predictions_new_label[0][0].lower() == ground_truth_name.lower():
                print(f"Predicted: {predictions_new_label[0][0].lower()} | Ground Truth: {ground_truth_name.lower()}")
                p_celebrity_list.append(predictions_new_label[0][1])
            else:
                p_celebrity_list.append(0)
    print('-------------------')
    print('Total number of images with no faces detected:', n_no_faces)           

    # save as excel file
    df=pd.DataFrame(predictions_list, columns=['top1','top2','top3','top4','top5'])
    # df.index=image_names
    df.index = valid_image_names

    df['p_celebrity_correct']=p_celebrity_list
    print('-------------------')
    print('Given face detected, the celebrity classification accuracy is:')
    
    # Calculate the number of non-zero and non-N values in p_celebrity_list and then divided by the number of non-N values.
    # print(sum([1 for p in p_celebrity_list if p != 0 and p != 'N']) / sum([1 for p in p_celebrity_list if p != 'N']))
    
    accuracy = sum([1 for p in p_celebrity_list if p != 0 and p != 'N']) / sum([1 for p in p_celebrity_list if p != 'N'])
    print(accuracy)

    if args.evaluation_metric:
        result_csv = f"{args.save_csv_path}/{args.evaluation_metric}.csv"
        df_result = pd.DataFrame([[args.cele_name, accuracy]], columns=["celebrity", "accuracy"])
        
        if os.path.exists(result_csv):
            df_existing = pd.read_csv(result_csv)
            df_combined = pd.concat([df_existing, df_result], ignore_index=True)
        else:
            df_combined = df_result

        df_combined.to_csv(result_csv, index=False)
        print(f"Appended result to {result_csv}")


    if args.save_excel_path is not None:
        os.makedirs(os.path.dirname(args.save_excel_path), exist_ok=True)

        df.to_excel(args.save_excel_path, index=True)
        
