Processed data is available at (https://drive.google.com/file/d/1Wvh_66fNIpybkEWVadcbqu0smWQiRSYh/view?usp=sharing).

To install the environment, the following commands can be used. ESM3 is open access but requires registering through huggingface hub.

mamba install pytorch torchvision pytorch-cuda=12.4 -c pytorch -c nvidia --yes
pip install esm wandb
mamba install ipykernel ipywidgets lightning biotite matplotlib seaborn --yes

train.py provides an entrypoint to train the model
