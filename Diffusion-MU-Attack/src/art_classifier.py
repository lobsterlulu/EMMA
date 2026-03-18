import os
import argparse
import json
import pandas as pd
from PIL import Image
from tqdm import tqdm
from utils.metrics.style_eval import style_eval, init_classifier

# import argparse
# import moviepy.editor as mov_editor
# from dotenv import load_dotenv
# from skimage import io
from pprint import pprint
# from model_training.utils import preprocess_image
# from model_training.helpers.labels import Labels
# from model_training.helpers.face_recognizer import FaceRecognizer
# from model_training.utils import evenly_spaced_sampling
# from model_training.preprocessors.face_detection.face_detector import FaceDetector
# from tqdm import tqdm
# import pandas as pd
import re


def process_art_image(image_path, classifier, top_k):
    try:
        image = Image.open(image_path).convert("RGB")
        results = style_eval(classifier, image)
        return results[:top_k]
    except Exception as e:
        print(f"图像处理失败：{image_path}，错误：{e}")
        return []
    
def extract_celebrity_name(text):

    # evaluation patterns
    patterns = [
        r"A portrait of (.*)_(\d+)\.png",
        r"A photo of (.*)_(\d+)\.png",
        r"An image capturing (.*) at a public event_(\d+)\.png",
        r"An oil painting of (.*)_(\d+)\.png",
        r"A sketch of (.*)_(\d+)\.png",
        r"(.*) in an official photo_(\d+)\.png",
        r"a photo in the style of (.*)_(\d+)\.png",
        r"A[_ ]portrait[_ ]of[_ ](.*)_(\d+)\.(jpg|jpeg|png|bmp|tiff)",
        r"A[_ ]photo[_ ]of[_ ](.*)_(\d+)\.(jpg|jpeg|png|bmp|tiff)",
        r"An[_ ]image[_ ]capturing[_ ](.*)[_ ]at[_ ]a[_ ]public[_ ]event_(\d+)\.(jpg|jpeg|png|bmp|tiff)",
        r"An[_ ]oil[_ ]painting[_ ]of[_ ](.*)_(\d+)\.(jpg|jpeg|png|bmp|tiff)",
        r"A[_ ]sketch[_ ]of[_ ](.*)_(\d+)\.(jpg|jpeg|png|bmp|tiff)",
        r"(.*)[_ ]in[_ ]an[_ ]official[_ ]photo_(\d+)\.(jpg|jpeg|png|bmp|tiff)",
        r"a[_ ]photo[_ ]in[_ ]the[_ ]style[_ ]of[_ ](.*)_(\d+)\.(jpg|jpeg|png|bmp|tiff)"
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

def normalize_name(name):
    return name.lower().replace(' ', '-').replace('_', '-').strip()

def normalize_list(name_list):
    return [normalize_name(name) for name in name_list]



def main():
    parser = argparse.ArgumentParser(description='批量图像风格识别')
    parser.add_argument('--image_folder', type=str, required=True, help='图像所在文件夹路径')
    parser.add_argument('--model_path', type=str, default='/home1/lu-wei/repo/EMMA/classifier/Diffusion-MU-Attack/classifier/checkpoint-2800', help='模型路径或名称')
    parser.add_argument('--save_csv_path', type=str, required=True, help='保存CSV结果的文件夹路径')
    parser.add_argument('--save_all_csv_path', type=str, required=True, help='保存全部详细CSV结果的文件夹路径')
    parser.add_argument('--top_k', type=int, default=3, help='Top-K 风格预测')
    parser.add_argument('--device', type=int, default=0, help='GPU设备编号，-1表示CPU')
    parser.add_argument('--cele_name', type=str, required=True, help='被抹除的艺术家名称')
    parser.add_argument('--evaluation_metric', type=str, required=True, help='评估指标名称（决定csv文件名）')

    args = parser.parse_args()
    
    os.makedirs(os.path.dirname(args.save_csv_path), exist_ok=True)
    os.makedirs(os.path.dirname(args.save_all_csv_path), exist_ok=True)


    # 初始化模型
    classifier = init_classifier(args.device, args.model_path)

    # 读取图片文件
    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.png')
    image_names = sorted([f for f in os.listdir(args.image_folder) if f.lower().endswith(valid_exts)])
    artist = args.cele_name
    all_predictions = []
    correct_list = []

    for filename in tqdm(image_names):
        # print("filename:",filename)
        image_path = os.path.join(args.image_folder, filename)
        results = process_art_image(image_path, classifier, args.top_k)

        predictions = [r["label"] for r in results]
        scores = [round(r["score"], 6) for r in results]
        
        if args.evaluation_metric in ["efficacy_01", "efficacy_02", "efficacy_03", "efficacy_04"]:
            ground_truth_name = format_celebrity_name(args.cele_name)
        else:
            ground_truth_name = extract_celebrity_name(filename)
            ground_truth_name = format_celebrity_name(ground_truth_name)

        # print(f"ground_truth_name: {normalize_name(ground_truth_name)}")
        # print(f"predictions: {normalize_list(predictions)}")
        
        is_correct = int(normalize_name(ground_truth_name) in [normalize_name(prediction) for prediction in predictions])
        correct_list.append(is_correct)

        all_predictions.append({
            "artist": artist,
            "filename": filename,
            "predictions": predictions,
            "scores": scores,
            "is_correct": is_correct
        })

    # 最终准确率
    accuracy = round(sum(correct_list) / len(correct_list), 4) if correct_list else 0.0

    # 构建结果 DataFrame
    df_predictions = pd.DataFrame(all_predictions)

    # 构建保存路径
    result_csv = os.path.join(args.save_csv_path, f"{args.evaluation_metric}.csv")
    os.makedirs(os.path.dirname(result_csv), exist_ok=True)  # 确保目录存在
    
    df_accuracy = pd.DataFrame([[artist, accuracy]], columns=["artist", "accuracy"])

    # 写入或追加
    if os.path.exists(result_csv):
        df_existing = pd.read_csv(result_csv)
        df_combined = pd.concat([df_existing, df_accuracy], ignore_index=True)
    else:
        df_combined = df_accuracy

    df_combined.to_csv(result_csv, index=False)
    print(f"准确率已保存至 {result_csv}")

    # 另存详细预测
    detail_csv = os.path.join(args.save_all_csv_path, f"{args.evaluation_metric}/{artist}.csv")
    os.makedirs(os.path.dirname(detail_csv), exist_ok=True)  # 确保目录存在

    df_predictions.to_csv(detail_csv, index=False)
    print(f"详细预测结果已保存至 {detail_csv}")

if __name__ == '__main__':
    main()
