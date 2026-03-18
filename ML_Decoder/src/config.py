import os
import sys
from os.path import dirname, abspath, join

ROOT = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(join(ROOT))

MACE_SAVE_RESULTS_DIR = join(ROOT, 'MACE/save_results')
MACE_SAVE_RESULTS_CSV_DIR = join(ROOT, 'MACE/save_results_csv')

UCE_SAVE_RESULTS_DIR = join(ROOT, 'UCE/generated_images')
UCE_SAVE_RESULTS_CSV_DIR = join(ROOT, 'UCE/save_results_csv')

ESD_SAVE_RESULTS_DIR = join(ROOT, 'erasing/generated_images')
ESD_SAVE_RESULTS_CSV_DIR = join(ROOT, 'erasing/save_results_csv')
