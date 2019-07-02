# PITL
Portable Image Translation Learning

<img width="591" alt="Screen Shot 2019-06-20 at 3 51 54 PM" src="https://user-images.githubusercontent.com/15677695/59886526-22274880-9374-11e9-93ec-6d661b32e29d.png">

## setup anaconda:

conda env create -f condaenv.yml

## example:

https://github.com/royerlab/pitl/blob/master/src/pitl/test/demo_fitl_2D.py

## gui:

https://github.com/royerlab/pitl/blob/master/src/pitl/gui/gui.py

## How to run same example with CLI

For now CLI can be only run as a module:

To run a Noise2Self:
```bash
python -m src.pitl.cli.cli noise2self 'relative/path/to/image'

```

To run a demo:
```bash
python -m src.pitl.cli.cli demo 2D

```

## Minimal Viable Product (MVP) statement:

- Python API that encapslates the internals
- CLI and GUI interface
- Self-contained executable (separate CLI and GUI)
- 2D and 3D data 
- Image translation: pair images (A,B):  A -> B (translation, denoising, etc...)
- Explicit self-supervised denoising (A): A -> A_denoised 
- Explicit noise 2 noise denoising
- 2D Multichannel
- Auto-tunning of receptive field based on autocorrelation

Extra:
- Isonet

## TODO:
  
- [ ] More tests
- [X] Download data for examples (as in CSBDeep)
- [ ] batch training for regression
- [ ] off-core feature storage 
- [ ] 1D version
- [ ] 2D multi-channel
- [ ] Put some work on choosing the right GPU and check that it is actually functional.
- [ ] Use the internal LightGBM  interface instead of the scikit-learn - like interface
- [ ] Compile LightGBM to the GPU. 
- [ ] Explore tuning of LightGBM (expose important parameters, automatic tuning?)
- [ ] self-contained executables (pyinstaller, use this as template: https://github.com/maweigert/spimagine/tree/master/build )
- [X] CLI -WIP 
- [ ] GUI


- [ ] More tests
- [X] Download data for examples (as in CSBDeep)
- [ ] batch training for regression
- [ ] off-core feature storage 
- [ ] 1D version
- [ ] 2D multi-channel
- [ ] Put some work on choosing the right GPU and check that it is actually functional.
- [ ] Use the internal LightGBM  interface instead of the scikit-learn - like interface
- [ ] Compile LightGBM to the GPU. 
- [ ] Explore tuning of LightGBM (expose important parameters, automatic tuning?)
- [ ] self-contained executables (pyinstaller, use this as template: https://github.com/maweigert/spimagine/tree/master/build )
- [X] CLI -WIP 
- [ ] GUI








