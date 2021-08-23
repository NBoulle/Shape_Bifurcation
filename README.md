# Installation

#### Install Firedrake
```
curl -O https://raw.githubusercontent.com/firedrakeproject/firedrake/master/scripts/firedrake-install
firedrake-install --doi 10.5281/zenodo.5217566
```

#### Activate Firedrake virtualenv
`source firedrake/bin/activate`

#### Install the Rapid Optimization Library
`pip install --no-cache-dir roltrilinos rol`

#### Install Fireshape
```
git clone git@github.com:fireshape/fireshape.git
cd fireshape
git checkout 2891d813f38ba69c6273ca312177deb5fdb0fe77
pip install -e .
```

# Run the code

Select one example of the paper in the folder Allen_Cahn, Navier_Stokes, or Hyperelasticity and run
`python3 main.py`

# Reference
```
@article{boulle2021control,
  title={Control of bifurcation structures using shape optimization},
  author={Boull{\'e}, Nicolas and Farrell, Patrick E and Paganini, Alberto},
  journal={arXiv preprint arXiv:2105.14884},
  year={2021}
}
```