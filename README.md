**Readme file for better understanding the code in this folder**


folder structure:

    folders
        - `configurations` : contains the `configurations.yaml` file for training and testing configurations
        - `data` : contains the scripts dealing with data (dataset, dataloader, transforms)
        - `demo_utils` : contains the scripts for testing the network giving a metadata image list and a  configuration
        - `model` : contains the script of the neural network
        - `state_dicts` : keeps the weights of the model 
        - `utils` : contains some utility file dealing with : loss, metrics, scheduler.
        - `old_utils_and_stuff` : useless scripts
    files:
        - `demo.py` : script for running a demo testing on images. Configured in demo_utils. The outputs are saved in the `outputs_demo` folder.
        - `test_environmental_creation.py` : utility for creating metadata for testing and checking the several environment of CuLane.
        - `test_environmental.py` : script for running the test on the environment of CuLane. The outputs are saved in the `outputs_test_environment` folder.
        - `ransac.py` : implementation of Ransac. Not used in this case.
        - `test.py` : script for running the test on the validation set of CuLane.
        - `train.py` : script for running the train on CuLane according to the `configurations.yaml` file.


*for running each script no parameters are required. Specify everything in the `configurations.yaml` file.*

