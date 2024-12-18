# MoDL - Quantum

PyTorch implementation of Quantum-Enhanced Model Based Deep Learning Architecture for Inverse Problems. Reuses code from https://github.com/bo-10000/MoDL_PyTorch, which is an unofficial reproduction of https://github.com/hkaggarwal/modl. Sincere thanks to both the authors for their significant contributions, their work has made this Quantum-Enhanced usecase possible.

Base MoDL Architecture:

![alt text](https://github.com/hkaggarwal/modl/blob/master/MoDL_recursive.png)

Slides covering Background, Motivation, Implementation and Results: https://docs.google.com/presentation/d/1fABv7CPdmYNZXOA3QEnzRPVBhy67kxGZ-eCryhOCtQs/edit?usp=sharing

## Primary References

1. MoDL: Model Based Deep Learning Architecture for Inverse Problems by H.K. Aggarwal, M.P Mani, and Mathews Jacob in IEEE Transactions on Medical Imaging,  2018 

Link: https://arxiv.org/abs/1712.02862

IEEE Xplore: https://ieeexplore.ieee.org/document/8434321/

2. Quanvolutional Neural Networks: Powering Image Recognition with Quantum Circuits. 2019., by Maxwell Henderson et al.

Link: https://arxiv.org/abs/1904.04767.

PennyLane Tutorial: https://pennylane.ai/qml/demos/tutorial_quanvolution. 

## Dataset

The multi-coil brain dataset used in the original paper has been made publically available by the authors of the MoDL Paper. It can be downloaded from the following link.

**Download Link** : https://drive.google.com/file/d/1qp-l9kJbRfQU1W5wCjOQZi7I3T6jwA37/view?usp=sharing

## Configuration file

The configuration files are in `config` folder. For the Base and Hybrid model, the K=1 config is trained for 50 epochs. The K=3 model is then trained for 10 epochs with the initial weights set to the final weights of the K=1 run.

For the Quantum model, weight transfer is still unsupported owing to some error. It is not a priority, since the K=1 run (set for 10 epochs) by itself incurrs a lot of computational overhead.

## Known Issues

Major Issue is with the computational overhead in running the Quantum_MoDL configuration. The only changes in implementation of the Quantum circuit between this and the Hybrid_MoDL are:

    i.  The Simulator used (`lightning.gpu` for Quantum, as opposed to `default.qubit` for Hybrid)
    ii. The Differentation Method used (`adjoint` for Quantum, as opposed to `backprop` for Hybrid)

It must be one of these changes that causes the computational overhead, and potentially other issues like the weight transfer issue. If one wishes to continue work on this codebase, it would be a good idea to examine this issue and identify a fix.

Apart from the above, there are still computational optimizations that can be leveraged, from JIT compilation, using JAX-supported Quantum devices, proper batch processing, etc. These changes will allow us to simulate Quanvolutional layer with higher number of qubits.

There is also scope for novel architectural developments, since the Quanvolutional Layer itself is pretty new by itself. It might be the case that certain architectures enable the Quanvolutional layer to learn features much better than what you would you expect with a traditional Convolutional layer for the same architecture. 

## Saved models

Saved models are not provided - the train script can be run with the dataset downloaded, with the config of your choice to generate the corresponding model in the `workspace` folder. 
