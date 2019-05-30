# PITL
Portable Image Translation Learning


## setup anaconda:

conda env create -f condaenv.yml

## example:

https://github.com/royerlab/pitl/blob/master/src/fitl/test/demo_fitl_2D.py

## Minimal Viable Product (MVP) statement:

- Python API that encapslates the internals
- CLI and GUI interface
- Self-contained executable (separate CLI and GUI)
- 2D and 3D data 
- Image translation: pair images (A,B):  A -> B (translation, denoising, etc...)
- Explicit self-supervised denoising (A): A -> A_denoised 
- 2D Multichannel
- Auto-tunning of receptive field based on autocorrelation

Extra:
- Isonet

## TODO:
  
- More tests
- Download data for examples (as in CSBDeep)
- batch training for regression
- off-core feature storage 
- 1D version
- 2D multi-channel
- Put some work on choosing the right GPU and check that it is actually functional.
- Use the internal LightGBM  interface instead of the scikit-learn - like interface
- Compile LightGBM to the GPU. 
- Explore tuning of LightGBM (expose important parameters, automatic tuning?)
- self-contained executables (pyinstaller, use this as template: https://github.com/maweigert/spimagine/tree/master/build )
- CLI 
- GUI








