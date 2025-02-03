mri - MRI Images
nonmri - CT Scan Images

Split - 80% Train, 20% Test
Split from 80% Train, 70% Train 25% Validation

Correct Amount:  60% 20% 20%

Changes: 

datasetBinaryMRI - All the images within the original folder was move up one hierarchy. Healthy and Tumor images were mixed into one folder.

The original names were MRI and CT
They were renamed to mri and nomri
Each contained 4000 images to predict their labels respectively.

datasetBinaryTumor - Original contained all the 8000 images. However CT scans were not required. They were deleted.

The Healthy and Tumor folders within the mri folder were renamed to notumor and tumor.

# Preprocessing

The images were relabelled into new classes
The images were resized to 224x224
The images were normalized
The images WERE NOT SCALED BECAUSE DATA AUGMENTATION WAS DONE

noise removal?
