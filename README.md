# celebA

The celebA dataset comprises over 200,000 jpeg images of various sizes, containing 10,000 different celebrities. Labels are provided for 40 facial attributes (hair style etc.) as well as bounding box coordinates for the faces.
The original compilers of the dataset used it to train a model to classify the attributes (Y. Sun, X. Wang, and X. Tang. "Deep learning face representation from predicting 10,000 classes". In CVPR, 2014.). Another interesting use was to use the detected attributes to score bounding box proposals for likelihood of containing a face (S. Yang, P. Luo, C. Change Loy and X. Tang. "From Facial Parts Responses to Face Detection: A Deep Learning Approach". arXiv:1509.06451).
Popular approaches for segmenting an image like the YOLO and Faster R-CNN models use very deep models and learn to classify the images in parallel with learning to draw boundary boxes.
Here I explore whether it's possible to perform rough boundary box regression alone on a scaled down version of this dataset with a smaller convNet of four layers.
