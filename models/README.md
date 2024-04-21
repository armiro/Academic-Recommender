### Model Directory
Model files downloaded from different source (i.e. `gensim` or `huggingface`) can be dumped as `pkl` file 
in this directory. `load_model` function in `utils/helper_functions.py` tries to save the downloaded model
as pickle file to this directory. It significantly increases model loading time.