```
                                             .--. .-..-..---.  .--. .---.  .--. 
                                            : .; :: :: :: .; :: ,. :: .; :: .; :
                                            :    :: :: ::   .': :: ::   .':    :
                                            : :: :: :; :: :.`.: :; :: :.`.: :: :
                                            :_;:_;`.__.':_;:_;`.__.':_;:_;:_;:_;

```

<div align="center">
<a href="https://github.com/juan-pineda"><img src="https://img.shields.io/apm/l/vim-mode" alt="website"/></a>
<a href="https://halley.uis.edu.co/"><img src="https://img.shields.io/static/v1?label=&labelColor=505050&message=website&color=%230076D6&style=flat&logo=google-chrome&logoColor=%230076D6" alt="website"/></a>
</div>

Python package for the creation of mock IFU observations out of numerical simulations of galaxies. Aurora can generate mock observations for H-alpha or HI using the whole physical and geometrical information of the particles in the simulation. Aurora works for simulations developed from the [RAMSES](https://bitbucket.org/rteyssie/ramses/src/master/) code or the [GADGET](https://wwwmpa.mpa-garching.mpg.de/gadget/) code. Therefore, this code can be easily adapted to work on other simulations with Adaptive Mesh Refinement (AMR) or for simulations with Smoothed Particle Hydrodynamics (SPH). 





## Installation 

Download the code by cloning the git repository using

```
$ git clone https://github.com/juan-pineda/Aurora.git
```

After downloading the code, you need to add the repository location address to the python path:

```
$ nano ~/.bashrc
```

Add to end of bashrc:

```
#aurora path 
export PYTHONPATH=${PYTHONPATH}:/home/user_name/dir/Aurora/
```

Update shell:

```
$ source ~/.bashrc
```

Then you can import Aurora as a usual library in Python:

```python
import aurora
```


## Usage ⚙️





## Documentation 📃
The documentation will be available in `/documentation` and in the <a href="https://github.com/juan-pineda/Aurora/wiki">wiki</a> of this repository.

## Requirements 🔨
* [Numpy](https://numpy.org/install/) version 1.14.2 or later
* [Astropy](https://www.astropy.org/) version 3.0.1 or later
* [Pympler](https://pympler.readthedocs.io/en/latest/) version 0.5 or later
* [Scipy](https://www.scipy.org/install.html) version 0.17 or later
* [Pynbody](https://pynbody.github.io/pynbody/installation.html) version 0.43 or later
* [Scikit_learn](https://scikit-learn.org/stable/install.html) version 0.19.1 or later

## Contact
Juan Carlos Basto Pineda - jcbastop@correo.uis.edu.co

## Contributors ✒️
* Juan Pineda [@juan-pineda](https://github.com/juan-pineda)
* Valentin Perret [@perretv](https://github.com/perretv)
* Benoît Epinat
* Philippe Amram
* Rolando Carvajal  [@rolando980709](https://github.com/rolando980709)
* Juan Manuel Pacheco [@JuanManuelPacheco](https://github.com/JuanManuelPacheco)
* Diego Rueda  [@druedaplata](https://github.com/druedaplata)

## License
 Pykinemetry is provided under "UIS" lincense.
