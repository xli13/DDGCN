# DD-GCN
Source code for "[A High Accuracy and Adaptive Anomaly Detection Model With Dual-Domain Graph Convolutional Network for Insider Threat Detection]"

# Environment Settings 
* python == 3.7   
* Pytorch == 1.1.0  
* Numpy == 1.16.2  
* SciPy == 1.3.1  
* Networkx == 2.4  
* scikit-learn == 0.21.3  

# Usage 
````
python main.py -d dataset -l labelrate
````
* **dataset**: including cert, required.  
* **labelrate**: including \[15, 30, 60\], required.  

e.g.  
````
python main.py -d cert -l 15
````

# Data
## Link
**Cert R4.2 & 6.2**: (https://kilthub.cmu.edu/articles/dataset/Insider_Threat_Test_Dataset/12841247)  
For graph date processing, we first process the original citesee dataset and replace associated fiels with ours.
For cert behavior data processing, please refer:
[D. C. Le, N. Zincir-Heywood, "Anomaly Detection for Insider Threats Using Unsupervised Ensembles," in IEEE Transactions on Network and Service Management, vol. 18, no. 2, pp. 1152–1164. June 2021, doi:http://doi.org/10.1109/TNSM.2021.3071928.]

## Usage
 The files in folders are as follows:
````
cert/
├─cert4.2.edge: edge file.  
├─cert4.2.feature: feature file.  
├─cert4.2.label: label file.  
├─testL/C.txt: test file. L/C, i.e., Label pre Class, L/C = 15, 30, 60.   
├─trainL/C.txt: train file. L/C, i.e., Label pre Class, L/C = 15, 30, 60.  
````
# Parameter Settings

Recorded in   **./DDGCN/config/[L/C][dataset].ini**  
e.g.   **./DDGCN/config/20citeseer.ini**  

* **Model_setup**: parameters for training DDGCN, such as nhid1, nhid2, beta, theta... 
* **Data_setting**: dataset setttings, such as paths for input, node numbers, feature dimensions...

# Reference
````
@ARTICLE{10044727,
  author={Li, Ximing and Li, Xiaoyong and Jia, Jia and Li, Linghui and Yuan, Jie and Gao, Yali and Yu, Shui},
  journal={IEEE Transactions on Information Forensics and Security}, 
  title={A High Accuracy and Adaptive Anomaly Detection Model With Dual-Domain Graph Convolutional Network for Insider Threat Detection}, 
  year={2023},
  volume={18},
  number={},
  pages={1638-1652},
  keywords={Behavioral sciences;Feature extraction;Convolutional neural networks;Topology;Adaptation models;Network topology;Couplings;Insider threat detection;anomaly detection;graph convolutional network},
  doi={10.1109/TIFS.2023.3245413}}

````
