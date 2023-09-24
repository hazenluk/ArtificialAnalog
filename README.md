# Realtime Modelling of Nonlinear Dynamical Systems for Analog Audio Emulation

An emulation of classic analog effects using machine learning and state-space analysis of nonlinear circuits (WIP).

[![Diode Clipper Distortion VST Example](Documentation/GitHub/youtube_thumbnail.png)](https://www.youtube.com/watch?v=AVldGj0ZkN4 "Diode Clipper Example")

The concept is based on the work of Julian Parker et. all [1], with new contributions being compilation of the model into a real-time performant VST and modelling of reponses to user parameter changes in dynamic use.

Training is currently ocurring using the guitar sample provided by Parker et. all, as this project is in its early stages. The sample is run through an LTSpice simulation to generate the target input, output, and state data. This will be replaced with a more robust training example set soon.

Current progress includes emulation of the results of Parker et. all in real-time and packaging into a VST. Future work will focus on the construction of a physical hardware device for sampling circuits, dynamic changes in the model in response to user input, improving model musicality, and publicly releasing a database of models emulating famous and classic circuits.

The proof of concept phase is already partially complete, but will be expanded to model more complicated behavior before moving on to data collection. The proof of concept model uses data generated from LTSpice simulations of circuit behavior as ground truth and learns an emulation of the circuit in a series of fully connected layers with hyperbolic tangent activations. Rather than treating audio as time series data and the circuit as a black box, the model learns to predict from the input and current model state (the voltage across all energy storage elements in the circuit) the output and next model state. This is a technically sufficient model of the circuit if it is considered time-invariant, since circuit output can then be completely characterized by current state and input. Time invariance is of little concern in the circuits of interest. Heating of elements over time could be considered a time-varying contributor to circuit behavior, but this could be modeled instead by considering temperature as a state variable. Currently, a functional diode clipper model has been implemented fully, from LTSpice data collection, to pytorch model training, to model encoding and optimization in a C++ based VST that can run 500 times faster than realtime on an AMD Ryzen 5 5600x.


## Building Project

This project depends on JUCE, pytorch, and RTNeural [2]. C++ code is built via CMake on Windows using VS2022, with project structure and CMakeLists based on GitHub user anthonyalfimov's JUCE-CMake-Plugin-Template [3]. JUCE should be automatically installed by CMake, and git should clone RTNeural as a submodule into the the Third_Party directory. Python version 3.11.5 was used during development, and the Python dependencies can be installed by running pip install -r requirements.txt from the Python directory.

## References

[1] https://www.dafx.de/paper-archive/2019/DAFx2019_paper_42.pdf

[2] https://github.com/jatinchowdhury18/RTNeural

[3] https://github.com/anthonyalfimov/JUCE-CMake-Plugin-Template
