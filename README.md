# The BSc thesis (Abstract)
"In the field of digital histopathology, computer- aided diagnosis of digitized tissue samples with computational algorithms is a rising research field. The tissue samples in this study are stained using chemicals that enhance the recognizability of different tissue structures. This staining can be highly variable, which has an impact on the performance of the computational algorithms. The aim of this project is to assess the use of three color normalization algorithms as a pre-processing step on the KI dataset from a collaborative research project between Karolinska Institutet and KTH Royal Institute of Technology. The color normalization algorithms aim to reduce the color variability of the data. The basis of the study is an implementation of the EfficentNet Convolutional Neural Network classification model, that was adapted for the specific needs of the study. Performance was assessed by firstly applying the color normalization filters to the dataset and training multiple models on each of the filtered datasets. The results from the individually trained models and the combined results with ensemble learning techniques were then analyzed. Our conclusions are clear, stain normalization filters significantly impacts classification performance metrics. The impact depends on the staining qualities of the filters. Ensemble learning techniques present a more robust performance than the individual filters with a performance comparable to the best performing filter."

Link to paper: https://drive.google.com/file/d/1k7QPwhpSuLVFdCFWhwGi5ycd2u-1sLhX/view 

# Acknowledgements
The computations and data handling were enabled by resources provided by the Swedish National Infrastructure for Computing (SNIC) at C3SE Chalmers University of Technol- ogy partially funded by the Swedish Research Council through grant agreement no. 2020/33-67. Research funding was also provided by ALF Medicine and SOF Clinical Odontological Research Funding. The authors would like to thank Karl Meinke and Rachael Sugars with team for their continued support and guidance during this project. 

# About
A repo for documentation Filip &amp; Albert.

A sheet detailing all experiments is available at this [link](https://docs.google.com/spreadsheets/d/1_sOiXOl1qm0wKVom49XMOGXiADfrzJuXoAEin4kzkko/edit#gid=1874826242).

# Setup 
(1) Document everything in the python-files (.py)  and (2) use a Jupiter Notebook parser library (https://pypi.org/project/p2j/) convert .py to .ipynb and (3) push changes to git.

# p2j documentation 
To create a jupyter notebook in terminal: "p2j train.py", if it's your second time creating this notebook use "p2j train.py -o" (-o for overwrite)

# Are you experiencing issues viewing .ipynb files?
GitHub sometimes has issues displaying .ipynb (Jupyter notebook files) if so, you could insted go to this page https://nbviewer.jupyter.org/ and paste in a .ipynb url in the dialog box, for example this one https://github.com/filipfusk/BscThesis/blob/main/parseData.ipynb. 
