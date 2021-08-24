
# Are socially-aware trajectory prediction models really socially-aware?

The official code for the paper: "Are socially-aware trajectory prediction models really socially-aware?", [Webpage](https://s-attack.github.io/), [arXiv]()

## _Absract_:

Our field has recently witnessed an arms race of neural network-based trajectory predictors.
While these predictors are at the core of many applications such as autonomous navigation or pedestrian flow simulations, their adversarial robustness has not been carefully studied.
In this paper, we introduce a socially-attended attack to assess the social understanding of prediction models in terms of collision avoidance. An attack is a small yet carefully-crafted perturbations to fail predictors. Technically, we define collision as a failure mode of the output, and propose hard- and soft-attention mechanisms to guide our attack. Thanks to our attack, we shed light on the limitations of the current models in terms of their social understanding.
We demonstrate the strengths of our method on the recent trajectory prediction models. Finally, we show that our attack can be employed to increase the social understanding of state-of-the-art models. 

Installation:
------------
Start by cloning this repository:
```
git clone https://github.com/vita-epfl/s-attack
cd s-attack
```

And install the dependencies:
```
pip install .
```
For more info on the installation, please refer to [Trajnet++](https://thedebugger811.github.io/posts/2020/03/intro_trajnetpp/)

## Dataset:
  
  * We used the trajnet++ [dataset](https://github.com/vita-epfl/trajnetplusplusdata/releases/tag/v4.0). For easy usage, we put data in DATA_BLOCK folder.
  
## Training/Testing:
In order to attack the model:
```
bash run.sh
``` 

## For citation:
```
@article{saadatnejad2021sattack,
  title={Are socially-aware trajectory prediction models really socially-aware?},
  author={Saadatnejad, Saeed and Bahari, Mohammadhossein and Khorsandi, Pedram and Saneian, Mohammad and Moosavi-Dezfooli, Seyed-Mohsen and Alahi, Alexandre},
  journal={arXiv preprint arXiv:},
  year={2021}
}
``` 