#### Project background

​   This project studies the local adversarial attack method in the black-box environment, and a new black-box attack method based on local interpretability model is proposed to generate adversarial examples, and is appied to different networks and datasets.

#### Environmental dependence

> Python 3.7.8

#### Standard library configuration

> numpy, tensorflow, pandas, matplotlib, scipy, tensorflow, torch, opencv-python, json, lime

#### Directory structure description
├── _pycache__                  // Project profile           
├── README.md              // Document description           
├── Load_Predict.py          // DataSet and Network Loading           
├── NES.py						// Gradient estimation algorithm           
├── CAM.py					  // CAM Important feature extraction algorithm           
├── attack.py					 // Black-FGSM&Black-PGD Algorithm flow         
└── main.py				                 

#### Instructions for use

- The main function (main) calls the Parser_Setting method to pre-define the hyperparameters, and then calls the attack_model class to select the network and model to be attacked. The available datasets are: Caltech101, Caltech256, ILSVRC2012. Target models are: Inception_v3, Xception, VGG16, ResNet50.
- In the attack_model class, the fgsm_model method and the pgd_model method are defined to carry out the FGSM attack and the PGD attack in the black box environment, respectively. The bool variable ismask is used to choose the local attack method or the global attack method.
- In addition, Gradient_estimation_fgsm and Gradiend_estimation_pgd methods are defined in NES.py to realize the details of black box attacks, including gradient estimation strategy and sample update strategy.
- After the confrontation sample is generated, the image matrix is saved in the .mat format. Then, the success rate of the attack, the L2 norm of the average disturbance, the peak signal-to-noise ratio of between the original sample and the confrontation sample, structural similarity, and other metrics are calculated to judge the advantages and disadvantages of the global attack and the local attack.
