# image-classification-port-to-pytorch
Ported an image classification CNN from tensorflow to pytorch

This notebook is my port of a final group ML project from my BU Masters program from tensorflow framework to pytorch. The project was graded well and the team happy with our academic result. We used Google Colab and A100 GPU compute and were instructed to use tensorflow as part of the assignment. However, after gradulation, I wanted to run the image classfication model locally with a new RTX 5090 GPU and tensorflow's drivers we're not yet upgraded that particular Nvidia GPU architecture. So, I ported the code to the PyTorch framework. 

The project was to build a CNN capable of image classification of the popular Food101 as hosted by huggingface.

IMPORTANT NOTES:
The main markdown represents the first project's results on tensorflow. So, there can be descrepancy between the stated results and the code processing results. The main point of this exercise was to ensure all features could be successful given the tech stack. 

A few additional features were created - a corrupt image detector, a class balance validator, some minor warning surpressors etc. 

The main ML run wrapper was from the course and did not need modification.
