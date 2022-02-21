# forest-prefetch

* ``code/`` contains the actual forest and tree code synthesizer discussed in the paper  
* ``data/`` contains scripts and files for running the experiments. Each folder represents one data set used in the experiments. There are a couple of scripts for convienience. Let ``dataset`` be a dataset of choice, then
    * ``dataset/init.sh`` can be used to download and prepare this dataset. Please note, that not all data-sets can be directly downloaded via script (``imdb``,``fact``,``trec``). Please download those manually. The URL can be found in the init-script. Also note, that ``wearable-body-postures`` needs some manual editing of the training data, because there is a wrong line in the original file.
    * ``dataset/trainForest.py`` This trains a new RF with 25 trees on the corresponding dataset using ``sklearn`` and stores the trained model as JSON file in ``dataset/text/forest_25.json``. Additionally, the model is exported as python pickle file in ``dataset/text/forsest_25.pkl``
    * ``generateCode.py`` This script does the actual code generation. It receives 2 parameters. The first parameter is dataset for which code should be generated, the second one is the target architecture (``arm`` or ``intel``). This will generate the necessary  cpp files for testing and generate a Makefile for compilation in the ``dataset/cpp/architechture/modelname `` folder. 
    * ``compile.sh`` This script receives two parameters. It will compile the cpp files for the given dataset (first parameter) and target architecture (second parameter). Please make sure, that the necessary compiler is installed on your system. For intel we use ``g++``. For arm ``arm-linux-gnueabihf-g++`` is used. 
    * ``run.sh`` This script receives two parameters. It will run the compiled cpp files for the given dataset (first parameter) and target architecture (second parameter). Results will be printed to std out. 
      ``runSKLearn.sh`` This script receives one parameter. It receives a folder and  will load the stored SKLearn model file (from the ``text`` folder) and run it on the corresponding dataset. Results will be printed to std out.
    * ``init_all.sh`` This will call the ``init.sh`` script on all folders
    * ``generate_all.sh`` This will call the ``generateCode.py`` script on all folders. It receives the target architecture as parameter
    * ``compile_all.sh`` This will call the ``compile.sh`` script on all folders.It receives the target architecture as parameter
    * ``run_all.sh`` This will call the ``run.sh`` script on all folders.It receives the target architecture as parameter


#How to

```bash
# Download the data
cd wine-quality
./init.sh

# Train the model
./trainForest.py

# Generate the code for intel
cd ..
./generatePrefetched.py wine-quality intel

# Compile it
./compile.sh wine-quality intel

# Run it
./collect.sh wine-quality intel
```