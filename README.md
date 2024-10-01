
# Code for PCSM

We provide the experimental code for Darcy Flow, Navier-Stokes, Airfoil and Plasticity.
Additional code will be provided in next version.

## Environment Installation

Create and activate an Anaconda Environment:
```
conda create -n PCSM python=3.8
conda activate PCSM
```

Install required packages with following commands:
```
pip install -r requirement.txt
```

## Data Preparation

Download the dataset from following links, and then unzip them in specific directory.
- Darcy Flow: [Google Driver](https://drive.google.com/file/d/1Z1uxG9R8AdAGJprG5STcphysjm56_0Jf/view?usp=sharing)
- Navier-Stokes: [Google Driver](https://drive.google.com/file/d/1lVgpWMjv9Z6LEv3eZQ_Qgj54lYeqnGl5/view?usp=sharing)
- Airfoil: [Google Driver](https://drive.google.com/drive/folders/1JUkPbx0-lgjFHPURH_kp1uqjfRn3aw9-?usp=sharing)
- Plasticity: [Google Driver](https://drive.google.com/file/d/14CPGK_ljae5c6dm2nRraY2kIDt39JX3d/view?usp=sharing)

## Experiment Running 

Run the experiments with following scripts.

- Darcy Flow: 
```
bash ./exp_scripts/darcy.sh [The Directory of Downloaded Data] 
# The Provided Directory should be like: XXX/Darcy
```

- Navier-Stokes:
```
bash ./exp_scripts/darcy.sh [The Directory of Downloaded Data]
# The Provided Directory should be like: XXX/NavierStokes
```

- Airfoil:
```
bash ./exp_scripts/ns.sh [The Directory of Downloaded Data]
# The Provided Directory should be like: XXX/naca
```

- Plasticity:
```
bash ./exp_scripts/plasticity.sh [The Directory of Downloaded Data]
# The Provided Directory should be like: XXX/plasticity
```

## Acknowledge

We thank following open-sourced projects, which provide the basis of this work.
- https://github.com/neuraloperator/neuraloperator
- https://github.com/gengxiangc/NORM
- https://github.com/thuml/Transolver
- https://github.com/nmwsharp/nonmanifold-laplacian
