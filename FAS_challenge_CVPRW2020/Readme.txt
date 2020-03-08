1. For Track1 Multi-modal:
Overall     -->   ACER=0.67
----------------------------------------
Prot. 4@1  (RGB+Depth+IR, separate 3 branch input)     -->   ACER=0.42
model1_2_pytorch/train_CDCN_3modality2_model2.py 
theta = 0.7
33 epochs
----------------------------------------
Prot. 4@2       -->   ACER=0.53
Score Fusion strategy:  0.87*((ModelA + ModelB)/2)+0.13*ModelC 
--
ModelA (Depth):   
model1_2_pytorch/train_CDCNpp_RGBorDepth_model1.py    
theta = 0.7
50 epochs
--
ModelB (RGB):   
model1_2_pytorch/train_CDCNpp_RGBorDepth_model1.py    
theta = 0.7
50 epochs
--
ModelC (RGB+Depth+IR, concat input 9 channels):   
model3_tensorflow/main.py
100,000 iterations

----------------------------------------
Prot. 4@3       -->   ACER=1.06
Score Fusion strategy:  0.8*((ModelA + ModelB)/2)+0.2*ModelC 
--
ModelA (Depth):   
model1_2_pytorch/train_CDCNpp_RGBorDepth_model1.py    
theta = 0.7
50 epochs
--
ModelB (RGB):   
model1_2_pytorch/train_CDCNpp_RGBorDepth_model1.py    
theta = 0.7
50 epochs
--
ModelC (RGB+Depth+IR, concat input 9 channels):   
model3_tensorflow/main.py
100,000 iterations



-----------------------------------------------------------
-----------------------------------------------------------

2. For Track2 Single-modal:
Overall     -->   ACER=4.81
----------------------------------------
Prot. 4@1       -->   ACER=6.74
model1_pytorch/train_CDCNpp_model1.py    
theta = 0.9
59 epochs
----------------------------------------
Prot. 4@2       -->   ACER=4.33
model1_pytorch/train_CDCNpp_model1.py    
theta = 0.5
50 epochs
----------------------------------------
Prot. 4@3       -->   ACER=3.61
model1_pytorch/train_CDCNpp_model1.py    
theta = 0.7
50 epochs
