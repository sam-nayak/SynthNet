# Fuzzy Deep Neural Networks

1. Membership Function Layer
2. Fuzzy Rule Layer
3. Input DR Layers
4. Fusion Layer
5. Fusion DR Layers
6. Task Driven Layer


## 1. Brief Description

* This is an implementation of Hierarchical Fused Deep Neural Network Model used for famous CIFAR-10 categorisation purpose.

## 2. Project Workflow
* Input layer starts with input vectors of an image of dimension (32x32x3), In this case there can be 32 vectors of size 32 for grey scale image and size 96 for coloured image.
* **Black** Each Input is connected with a membership function defined by:
    > ![Membership Function]
* Outputs of all membership functions are connected together using AND operation by:
    > ![Fuzzy Rule]
* **Blue** Now, all your Inputs are Densely connected by (blue) Input DR Layer using Sigmoid activation, by following formula:
    > ![Input DR]
    * There can be multiple Input DR Layers, we have used 2 layers as shown in the paper.
* **Green** Finally output of `Fuzzy Rule Layer` & `Input DR Layer` are fused together.
* Now, the fused output is passed through `Fusion DR Layer`. `Note: These can be multiple layers, we used single layer in this case`. Following formula is used to compute these layers:
    > ![Fusion DR]
* **Red** The Final `Task Driven Layer` is the classification layer. That classifies using softmax function that is as follows:
    > ![Task Driven]
* After Classification loss is calculated using mean-square-loss `mse`.
    > ![MSE]
## 3. Running the project

Create virtual env and install the requirements.

```commandline
python run.py --h
usage: run.py [-h] [--learning-rate LEARNING_RATE] [--epoch EPOCHS]
              [--batch-size BATCH_SIZE] [--colour-image]
              [--membership-layer-units MEMBERSHIP_LAYER_UNITS]
              [--first-dr-layer-units DR_LAYER_1_UNITS]
              [--second-dr-layer-units DR_LAYER_2_UNITS]
              [--fusion-dr-layer-units FUSION_DR_LAYER_UNITS]
              [--hide-graph]

FuzzyDNN on CIFAR-10

optional arguments:
  -h, --help            show this help message and exit
  --learning-rate LEARNING_RATE
                        Learning Rate of your classifier. Default 0.0001
  --epoch EPOCHS        Number of times you want to train your data. Default
                        100
  --batch-size BATCH_SIZE
                        Batch size for prediction. Default=16.
  --colour-image        Passing this argument will keep the coloured image
                        (RGB) during training. Default=False.
  --membership-layer-units MEMBERSHIP_LAYER_UNITS
                        Defines the number of units/nodes in the Membership
                        Function Layer
  --first-dr-layer-units DR_LAYER_1_UNITS
                        Defines the number of units in the first DR Layer
  --second-dr-layer-units DR_LAYER_2_UNITS
                        Defines the number of units in the second DR Layer
  --fusion-dr-layer-units FUSION_DR_LAYER_UNITS
                        Defines the number of units in the Fusion DR Layer
  --fusion-dr-layer-units FUSION_DR_LAYER_UNITS
                        Defines the number of units in the Fusion DR Layer
  --hide-graph          Hides the graph of results displayed via matplotlib

example usage:
  run.py
  run.py --epoch 100 --batch-size 8 --learning-rate 0.001 --membership-layer-units 256 --first-dr-layer-units 128 --second-dr-layer-units 64
```

[Membership Function]: ./formulas/MembershipFunction.png
[Fuzzy Rule]: ./formulas/FuzzyRuleLayer.png
[Input DR]: ./formulas/InputDR.png
[Fusion DR]: ./formulas/FusionDR.png
[Task Driven]: ./formulas/TaskDriven.png
[MSE]: ./formulas/MSE.png
