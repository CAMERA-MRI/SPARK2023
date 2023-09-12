# Challenge Overview
**Gliomas**

- gliomas = among deadliest types of cancer, survival rate less than 2 years after diagnosis
- challenging to diagnose, hard to treat and inherently resistant to conventional therapy
- patient survival largely influenced by molecular, genetic and clinical factors including tumor volume, tumor grade, age at diagnosis and histologic findings
- detection and analysis dependent on: identifying appropriate pathological features on brain magnetic resonance imaging (MRI) and confirmation by histopathology evaluation of biopsied tissue
- CAD using ML holds promise in increasing accuracy of tumor diagnosis, early detection, classification and in predicting tumor recurrence and patient survival

**Africa-specific**

- years of research have decreased mortality rates across the Global North, while chances of survival among individuals in low- and middle-income countries (LMICs) remain unchanged and are significantly worse in Sub-Saharan Africa (SSA) populations
- unclear if the state-of-the-art methods (for detecting, characterising and classifying gliomas) can be widely implemented in SSA given the extensive use of lower-quality MRI technology which produces poor contrast and resolution
- also propensity for late presentation of disease at advanced stages as well as the unique characteristic of gliomas in SSA
- research innovations have not translated to survival improvements in LMICs, particularly African populations
- death rates from glioma in SSA are among the highest in the world and continue to rise
- death rates in SSA rose on average by ~25% while they dropped in Global North by 10-30% from 1990 to 2016
- disparity could be due to several factors including:
- delayed presentation
- high incidence of infectious disease comorbidities e.g. HIV
- sever shortage of healthcare infrastructure
- lack of skilled expertise in diagnosis and treatment
- novel use of ML will further widen survival disparities if SSA populations are not included in innovative solutions that collectively benefit all patients
- ML can close survival disparity gaps by overcoming challenges in low-resourced settings where time consuming manual evaluations are limited to the rare centers in urban areas that can afford highly skilled expert personnel to perform tumor analysis
- unclear if state-of-the-art ML methods developed using BraTS data can be widely implemented for clinical use in SSA particularly given the extensive use of lower quality MRI technology in the region
- MRI in SSA typically have poor image contrast and resolution and may require further advanced pre-processing to enhance their resolution before application of ML methods
    
**Most typical errors observed in previous BraTS instances:**

- extension of ET vessels and choroid plexus
- under and over segmented areas of SNFH
- incorrect segmentation of areas of hemorrhage as ET

## Data

- data sets from adult populations collected through a collaborative network of imaging centres in Africa
- collection of pre-operative glioma data comprising of multi-parametric (mpMRI) routine clinical scans acquired as part of standard clinical care from multiple institutions and different scanners using conventional brain tumor imaging protocols
- differences in imaging systems and variations in clinical imaging protocols = vastly heterogeneous image quality
- ground truth annotations were approved by expert neuroradiologists and reviewed by board-certified radiologists with extensive experience in the field of neuro-oncology
**training (70%), validation (10%), testing (20%)**
- training data provided with associated ground truth labels, and validation data without any associated ground truth
- image volumes of:
    - T1-weighted **(T1)**
    - post gadolinium (Gd) contrast T1-weighted **(T1Gd)**
    - T2-weighted **(T2)**
    - T2 Fluid Attenuated Inversion Recovery **(T2-FLAIR)**
    
## Standardised BraTS **pre-processing workflow** used

- identical with pipeline from BraTS2017-2022 - publicly available
- conversion of DICOM to files to NIfTI file format --  strips accompanying metadata from the images and removes all Protected Health Information from DICOM headers
- cor-registration to the same anatomical template (SRI24)
  - resampling to uniform isotropic resolution (1mm^2)
  - skull stripping (uses DL approach) --  mitigates potential facial reconstruction/recognition of the patient
        
## Generation of **ground truth labels**

- volumes segmented using STAPLE fusion of previous top-ranked BraTS algorithms (nnU-Net, DeepScan and DeepMedic)
- segmented images refined manually by volunteer trained radiology experts of varying rank and experience
- then two senior attending board-certified radiologists with 5 or more years experience reviewed the segmentations
- iterative process until the approvers found the refined tumor sub-region segmentations accceptable for public release and challenge conduction
- finally approved by experienced board-certified attending neuro-radiologists with more than 5 years experience in interpreting glioma brain MRI
    
- **sub-regions** -- these are image-based and do not reflect strict biologic entities
    - enhancing tumor (ET) = tumor segments exhibiting a discernible rise in T1 signal on post-contrast images compared to pre- 
      contrast images
    - non-enhancing tumor core (NETC) = This classification comprises all segments of the tumor core (the area typically removed by 
      a surgeon) that show no enhancement.
    - surrounding non-enhancing flair hyperintensity (SNFH) = This refers to the complete extent of FLAIR signal abnormality 
      surrounding the tumor, which excludes any regions that are part of the tumor core. 

    Therefore, the sub-regions need to be converted back to the original classes:
    - NCR = necrotic tumor core
    - ED = peritumoral edematous tissue
    - ET = enhancing tumor

## Important information
### Training & Val
- training data has ground truths available
- validation data (released 3 weeks after training data) does not have any ground truth
- ***NB: challenge participants can supplement the data set with additional public and/or private glioma MRI data for training algorithms***
    - supplemental data set must be explicitly and thoroughly described in the methods of submitted manuscripts
    - required to report results using only the BraTS2023 glioma data and results that include the supplemental data and discuss potential result differences
- for submission participants are required to package their developed approach in an MLCube container following provided in the Synapse platform - Cube containers are automatically generated by GaNDLF and will be used to evaluate all submissions through the MedPerf platform

#### Evaluation metrics
- Dice Similarity Coefficient  
- 95% Hausdorff distance (as opposed to standard HD in order to avoid outliers having too much weight)
- precision (to complement sensitivity)

#### Other
- submitted algorithms will be ranked based on the generated metric results on the test cases by computing the summation of their ranks across the average of the metrics described above as a univariate overall summary measure
- outcome will be plotted via an augmented version of radar plot - to visualise the results
- missing results on test cases or if an algorithm fails to produce a result metric for a specific test case the metric will be set to its worst possible value (e.g. 0 for DSC) 
- uncertainties in rankings will be assessed using permutational analyses“Performance for the segmentation task will be assessed based on relative performance of each team on each tumor tissue class and for each segmentation measure
- multiple submissions to the online evaluation platforms will be allowed 
- top ranked teams in validation phase will be invited to prepare their slides for a short oral presentation of their method during the BraTS challenge at MICCAI 2023
- “all participants will be evaluated and ranked on the same unseen testing data, which will not be made available to the participants, after uploading their containerized method in the evaluation platforms
- final top ranked teams will be announced at MICCAI 2023 (monetary prizes)
- participating African teams with best rank will receive Lacuna Equity & Health Prizes (limited to BraTS-Africa BrainHack teams)