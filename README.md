Three codes are presented:
- compte_featres.py: Obtain a file containing the feature vectors of every image of a dataset.
- compute_pres.py: Calculate the presicions for every image of a dataset as query, obtains AP and prints mAP, plots presicion-recall curve.
- best_worst_retrievals.py: Plot a graph of the 5 best or worst retrieval results.

For each code, to run, the model and dataset to use must be defined along with the path to it in the DATASET, MODEL and data_dir variables. All required libraries are imported, and can be installed through pip as normal, (pip install torch torchvision scikit-image numpy matplotlib pillow) For clip, one must execute: pip install git+https://github.com/openai/CLIP.git.

The valid values for the DATASET and MODEL variables are:

Datasets:
- 'simple1k'
- 'Paris_val'
- 'VOC_val'

Models:
- 'resnet18'
- 'resnet34'
- 'clip'
- 'dinov2'

compute_features.py saves a numpy file to a path that can be modified in line 69. Both compute_pres.py and best_worst_retrievals.py generate graphs and save them as images on a path that can also be modified on 'plt.savefig(...)'
