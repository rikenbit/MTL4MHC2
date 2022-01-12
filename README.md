# MTL4MHC2
#### We suggest applying the multi-task learning for MHC class II-antigen binding prediction. It was reported in [MHC class II binding prediction by using the multi-task learning](URL)
---


# Dataset 
 
#### We used the MHC class I/II datasets curated by Gopalakrishnan Ven-katesh et al. [(Venkatesh et al., 2020)](https://github.com/gopuvenkat/MHCAttnNet) for deep learning model investiga-tion. Splited data, which was used for evaluation, is available from [here](https://bioinformatics.riken.jp/MTL4MHC2/evaluation_dataset_5cross/).
 
#### Class II MHC dataset by HLA subclass: For model evaluations, we used the MHC class II dataset, which was collected from the IEDB (Sahin et al., 2017) (as of 2021). This dataset contains 131060 classical MHC gene alleles. We labeled binding infor-mation with IC50 values of <1000 nM. This dataset is uvailable from [here](https://bioinformatics.riken.jp/MTL4MHC2/evaluation_dataset_IEDB/).

---

# Model and Weight 
 
#### Downsampling test models are available from [here](https://bioinformatics.riken.jp/MTL4MHC2/model/downsampling/). 
 
#### Bi-LSTM model, Multi-head Bi-LSTM model, Multi-task Bi-LSTM model are available from [here](https://bioinformatics.riken.jp/MTL4MHC2/model/).
 

---

# Script 

#### We open our model screening  scripts and main model creation scripts respectively. [here]() 


---

# Evaluation data 

#### We open our evaluation results as the jupyter notebook file. [here]() 

#### MTL4MHC2_model_evaluation_1.ipynb : The evaluation result of dataset1. 

#### MTL4MHC2_evaluation2_prediction.ipynb : The prediction script of dataset2. 

#### MTL4MHC2_evaluation2_result.ipynb : The evaluation result of dataset2. 








