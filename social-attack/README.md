## Social-attack
### Are socially-aware trajectory prediction models really socially-aware?

The official code for the paper: "Are socially-aware trajectory prediction models really socially-aware?", [Webpage](https://s-attack.github.io/), [arXiv](https://arxiv.org/abs/2108.10879)

&nbsp;


#### Installation:

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

#### Dataset:
  
  * We used the trajnet++ [dataset](https://github.com/vita-epfl/trajnetplusplusdata/releases/tag/v4.0). For easy usage, we put data in DATA_BLOCK folder.
  
#### Training/Testing:
In order to attack the LSTM-based models (S-lstm, S-att, D-pool):
```
bash lrun.sh
```
In order to attack the GAN-based models:
```
bash grun.sh
```

### For citation:
```
@article{saadatnejad_sattack,
  title = {Are socially-aware trajectory prediction models really socially-aware?},
  journal = {Transportation Research Part C: Emerging Technologies},
  volume = {141},
  pages = {103705},
  year = {2022},
  issn = {0968-090X},
  doi = {https://doi.org/10.1016/j.trc.2022.103705},
  author = {Saeed Saadatnejad and Mohammadhossein Bahari and Pedram Khorsandi and Mohammad Saneian and Seyed-Mohsen Moosavi-Dezfooli and Alexandre Alahi},
}

``` 
