Predicting the Severity of Postoperative Scars Using Artificial Intelligence Based on Images and Clinical Data 
==========================
![image](https://user-images.githubusercontent.com/79613225/198456405-d08ebb30-443f-4cdd-b3ab-0fc68400c801.png)

### Abstract
Evaluation of scar severity is crucial for determining proper treatment modalities; however, there is no gold standard for assessing scars. This study aimed to develop and evaluate an artificial intelligence model using images and clinical data to predict the severity of postoperative scars. This study revealed that a deep neural network model derived from image and clinical data could predict the severity of postoperative scars. The proposed model may be utilized in clinical practice for scar management, especially for determining severity and treatment initiation.
<br><br/>

### Results
|                   |  accuracy  |   AUC   |
|-------------------|------------|---------|
|    image based    |     72.5   |  93.1   |
|clinical-data based|     69.2   |  90.5   |
|     combined      |     72.9   |  93.8   |

![image](https://github.com/L-YUNNA/Scar-Severity-Prediction-pytorch/assets/129636660/b63f55d7-5c80-4e8b-90e8-54dd15d7c87f)


## Requirement
- Windows10, 3*RTX A4000, PyTorch 1.9.0, CUDA 11.2 + CuDNN 8.1.0, Python 3.9   
<br><br/>
   
## Usage
  
  
      # train
      python train_clinical.py --ngpu 3 --epochs 50 --batch-size 16 --lr 0.01 --momentum 0.1 --weight-decay 1e-5 --seed 1177 --prefix clinical_checkpoint ./data/clinical
      python train_image.py --ngpu 3 --epochs 100 --batch-size 16 --lr 0.01 --momentum 0.1 --weight-decay 1e-5 --kfold 5 --att-type CBAM --prefix image_checkpoint ./data/image
      python train_combined.py --ngpu 3 --epochs 100 --batch-size 16 --lr 0.01 --momentum 0.1 --weight-decay 1e-5 --seed 1177 --kfold 5 --att-type CBAM  --prefix combined_checkpoint ./data/combined
      

<br><br/>

## Reference
- **CBAM: Convolutional Block Attention Modul**
- Link: https://github.com/Jongchan/attention-module
