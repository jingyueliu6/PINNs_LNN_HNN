# Physics-informed Neural Networks to Model and Control Robots:a Theoretical and Experimental Investigation

## Abstract
This work concerns the application of physics-informed neural networks to the modeling and control of complex robotic systems. Achieving this goal required extending Physics Informed Neural Networks to handle non-conservative effects. We propose to combine these learned models with model-based controllers originally developed with first-principle models in mind. By combining standard and new techniques, we can achieve precise control performance while proving theoretical stability bounds. These validations include real-world experiments of motion prediction with a soft robot and of trajectory tracking with a Franka Emika manipulator.

## Paper
This repository contains the simulations parts as presented in the paper Physics-informed Neural Networks to Model and Control Robots:a Theoretical and Experimental Investigation. The paper's link is [Physics-informed Neural Networks to Model and Control Robots:
a Theoretical and Experimental Investigation](https://arxiv.org/pdf/2305.05375.pdf).

## requirements
jax                       0.3.4 

jaxlib                    0.3.0+cuda11.cudnn82 

(
jax                       0.3.25 

jaxlib                    0.3.25+cuda11.cudnn82 
(for the franka example)
)

## Data
The data utilized for training the models is available at: [link to data] (https://drive.google.com/drive/folders/1mzEgNQt-V5AKUr12cnlKj7_jbXqZndC5?usp=drive_link). This link also include some testing datasets. 

Regarding the one-segment spatial robot, no specific generated dataset exists for its testing. Instead, the testing has been conducted by manually selecting initial states and random inputs shown in the code. The results are then compared directly with the corresponding model in MATLAB. About the one-segment planar robot, the test code incorporates an analytical dynamics model. You can select appropriate initial states and inputs to assess its performance.


