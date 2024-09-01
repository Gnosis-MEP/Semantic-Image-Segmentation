# Semantic Image Segmentation
Provides Semantic Image Segmentation

# Installation

## Configure .env
Copy the `example.env` file to `.env`, and inside it replace the variables with the correct information.


## Installing Dependencies

### Using pip
Load the environment variables from `.env` file using `source load_env.sh`.

To install from the `requirements.txt` file, run the following command:
```
$ pip install -r requirements.txt
```

# Running
Enter project python environment (virtualenv or conda environment)

**ps**: It's required to have the .env variables loaded into the shell so that the project can run properly. An easy way of doing this is using `pipenv shell` to start the python environment with the `.env` file loaded or using the `source load_env.sh` command inside your preferable python environment (eg: conda).

Then, run the the component with:
```
$ ./semantic_image_segmentation/OI_mask_creator.py <input_dir> <sub_dataset_id> [<masked_class1>,<masked_class...N>]
```

Example:
```
$ ./semantic_image_segmentation/OI_mask_creator.py ../Datasets/HS-D-B-1 HS-D-B-1-10S car,person
```

Outputs generated at:

* Masks: `./outputs/masks/<sub_dataset_id>/<1stOIclass>_<2ndOIclass>/mask_frame_*.png`
* OI containing Samples:  `./outputs/samples/<sub_dataset_id>/<1stOIclass>_<2ndOIclass>/frame_*.png`