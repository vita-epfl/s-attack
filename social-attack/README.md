# Are socially-aware trajectory prediction models really socially-aware?


![pull figure](../docs/pull.png)

In this project, you can find the official codes of the paper and instructions on how to run them. The codes are in python.




## Installation

Start by cloning this repository:
```
git clone https://github.com/vita-epfl/s-attack
cd s-attack/social-attack
```

And install the dependencies:
```
pip install .
```
For more info on the installation, please refer to [Trajnet++](https://thedebugger811.github.io/posts/2020/03/intro_trajnetpp/)

## Dataset
  
  * We used the trajnet++ [dataset](https://github.com/vita-epfl/trajnetplusplusdata/releases/tag/v4.0). For easy usage, we put data in DATA_BLOCK folder.
  
## Training/Testing
In order to attack the LSTM-based models (S-lstm, S-att, D-pool):
```
bash lrun.sh
```
In order to attack the GAN-based models:
```
bash grun.sh
```

----
### For citation:
```
@article{saadatnejad2022sattack,
     author = {Saeed Saadatnejad and Mohammadhossein Bahari and Pedram Khorsandi and Mohammad Saneian and Seyed-Mohsen Moosavi-Dezfooli and Alexandre Alahi},
     title = {Are socially-aware trajectory prediction models really socially-aware?},
     journal = {Transportation Research Part C: Emerging Technologies},
     volume = {141},
     pages = {103705},
     year = {2022},
     issn = {0968-090X},
     doi = {https://doi.org/10.1016/j.trc.2022.103705},
}

``` 
