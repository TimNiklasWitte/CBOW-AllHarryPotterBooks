# CBOW: Continuous Bag-of-Words

A CBOW model was trained on [all harry potter books](https://www.kaggle.com/datasets/moxxis/harry-potter-lstm).


## Evaluation

### Loss

<img src="./plots/Loss.png" width="350" height="250">

### Latent space

<img src="./media/PointsOnly.png" width="600" height="400">

<img src="./media/Words.png" width="600" height="400">

<img src="./media/Magic.png" width="600" height="400">

## Merge saved_models zip files into a single zip file

```
zip -F saved_models.zip --out single_saved_models.zip
```