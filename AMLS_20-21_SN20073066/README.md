# AMLS_assignment20_21
This repo provides the solutions for AMLS project A1,A2,B1 and B2

## Library and enviroment
OpenCV, Scikit-Learn, Pandas, Joblib, Tensorflow 1.14, CUDA, Python 3.7

## File structure
- AMLS_20-21_SN20073066
  - A1
    - gender.py (will be included in main.py)
    - gende_LR.ipynb (training code contains model construction and parameter tuning)
  - A2
    - model (contains pre-trained CNN)
    - smiling.py (will be included in main.py)
    - smiling.ipynb (training code contains model construction and parameter tuning)
  - B1
    - faceshape.m (pre-trained SVM)
    - face_shape.py (will be included in main.py)
    - SVM_face_shape.ipynb (training code contains model construction and parameter tuning)
  - B2
    - model (contains pre-trained CNN)
    - eye_color.py (will be included in main.py)
    - eye_color.ipynb (training code contains model construction and parameter tuning)
  - Datasets
  - main.py (display all model performances)
  - README.md
## Reminder
- use Tensorflow 1.14 is preferrable
- main.py can run in both terminal and jupyter notebook. In jupyter notebook use following command:
  > %run main.py
- before run main.py, remember to decompress zip files in A2,B1,B2
