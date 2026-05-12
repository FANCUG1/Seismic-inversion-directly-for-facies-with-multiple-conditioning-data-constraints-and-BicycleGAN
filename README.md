# Seismic-inversion-directly-for-facies-with-multiple-conditioning-data-constraints-and-BicycleGAN

Seismic inversion directly for facies with multiple conditioning data constraints and BicycleGAN

Wenyao Fan, Xuechao Wu, Shijie Peng, Gang Liu, Qiyu Chen, Yang Li, Leonardo Azevedo

1. DER/CERENA, Instituto Superior Técnico, Universidade de Lisboa, Lisboa, Portugal
2. School of Computer Science, China University of Geosciences, Wuhan 430074, China
3. School of Artificial Intelligence and Computer Science, Hubei Normal University, Huangshi, 435002, China
4. Hubei Post and Telecommunications Planning and Design Co., Ltd, Wuhan 430074, China

One should prepare the python packages before running these codes:
1. The main installed packages are: 
Python 3.9.7, PyTorch 1.10.1, NumPy 1.23.5, openCV >= 4.5.5 and SciPy 1.9.3

2. After installing these python packages successfully, one should firstly prepare the original training dataset, which will be saved into the Dataset_syn/Dataset_Norne, including
Facies_TI and Seismic_TI. These prior training samples can be obtained through running Create_Facies_and_Seismic_TI.py

3. After preparing the training dataset, one can directly run the train.py to train the neural network, and the hyperparameters can be fine-tuned in the train.py file.
The pretrained network, including Generator, two Discriminator, and Encoding function, will be saved into the weight document.
Meanwhile, after per epoch is finished, one can observe the modeling results based on one testing samples.
   
