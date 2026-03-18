python -m main/clip_classifier/classify/inference_images.py --input_folder /home1/lu-wei/repo/MACE/save_results/nsfw/generated_images/harassment/efficacy_01/erased --output_folder /home1/lu-wei/repo/MACE/save_results_csv/image_info/nsfw --gpu 0



python -m main.clip_classifier.classify.inference_images \
  --input_folder /home1/lu-wei/repo/MACE/save_results/nsfw/generated_images/harassment/efficacy_01/erased \
  --output_folder /home1/lu-wei/repo/MACE/save_results_csv/image_info/nsfw/harassment \
  --gpu 1

python -m main.clip_classifier.classify.inference_images \
  --input_folder /home1/lu-wei/repo/MACE/data/nsfw/shocking/shocking \
  --output_folder /home1/lu-wei/repo/MACE/save_results_csv/image_info/nsfw/test_shocking \
  --gpu 1

python -m main.clip_classifier.classify.inference_images \
  --input_folder /home1/lu-wei/repo/MACE/data/nsfw/self-harm/self-harm\
  --output_folder /home1/lu-wei/repo/MACE/save_results_csv/image_info/nsfw/test_self-harm \
  --gpu 1


  /home1/lu-wei/repo/MACE/data/nsfw/self-harm/self-harm


python main/clip_classifier/classify/inference_images.py --input_folder <path/to/images> --output_folder <path/output>

/home1/lu-wei/repo/MACE/save_results/nsfw/generated_images/sexual/efficacy_02/erased

python -m main.clip_classifier.classify.inference_images \
  --input_folder /home1/lu-wei/repo/MACE/save_results/nsfw/generated_images/sexual/efficacy_02/erased \
  --output_folder /home1/lu-wei/repo/MACE/save_results_csv/image_info/nsfw/sexual/efficacy_02 \
  --gpu 1

python -m main.clip_classifier.classify.inference_images \
  --input_folder /home1/lu-wei/repo/MACE/save_results/nsfw/generated_images/sexual/efficacy_03/erased \
  --output_folder /home1/lu-wei/repo/MACE/save_results_csv/image_info/nsfw/sexual/efficacy_03 \
  --gpu 1

  
python -m main.clip_classifier.classify.inference_images \
  --input_folder /home1/lu-wei/repo/MACE/save_results/nsfw/generated_images/sexual/generality/erased \
  --output_folder /home1/lu-wei/repo/MACE/save_results_csv/image_info/nsfw/sexual/generality \
  --gpu 1