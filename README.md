# Bass Tablature Accompaniment Generation

This is the accompanying repository of the following paper:

> Anoufa, O., D'Hooge, A., Déguernel, K. "Conditional Generation of Bass Guitar Tablature for Guitar Accompaniment in Western Popular Music", AI Music Conference 2025. 

If you use any of this code, please cite the paper. You can for instance use the following bibtex entry:

```
@inproceedings{anoufaConditional2025,
  title = {Conditional {{Generation}} of {{Bass Guitar Tablature}} for {{Guitar Accompaniment}} in {{Western Popular Music}}},
  booktitle = {{{AI Music Creativity Conference}} ({{AIMC}})},
  author = {Anoufa, Olivier and D'Hooge, Alexandre and D{\'e}guernel, Ken},
  year = {2025}
}
```

Please note that this code is shared under a GPL3 License. Refer to the `LICENSE` file for details.
Also be aware that this code is heavily based on the following repository: [https://github.com/melkor169/CP_Drums_Generation](https://github.com/melkor169/CP_Drums_Generation).
We advise also citing the paper corresponding to the original repository if the code is used in research work.

## Demo Website

If you want to access the demonstration website of the generated files, go to:

[https://adhooge.github.io/BassTablatureGeneration/](https://adhooge.github.io/BassTablatureGeneration/)

## Configuration

You can use the provided `environment.yml` file to set up a Conda environment:
```
conda env create -f environment.yml
```
Some packages can be changed but the versions of numpy and tensorflow should be maintained to ensure correct run of the model.

## Data

This work is based on the DadaGP dataset ([Sarmento et al. 2021](https://zenodo.org/records/5624597)). 
To remove the need to recompute computationally intense rhythm guitar identification and token extraction, we 
directly share the pre-processed dataset.

The train/validation/test pickle files are available in the `data.zip` archive. 
You will have to **unzip** it before using the code.
Were the pickle files to grow out of date, or a need for the full data without splits, feel free to contact us at the email address shared on the paper.

The data was extracted from the original DadaGP dataset that is restricted to research purposes (unlicensed).
The data files of this repository are shared under an ODbL license. Refer to the `DATA_LICENSE` file for details.

## Training

You can reproduce the model's training after unzipping data with the `src/model_train.py` script. 
It requires a GPU device to run in a decent amount of time, which is still several hours.

## Inference

You can use your trained model or the checkpoint provided in `ckpt/` to generate samples from the test set.

## Issues

If you encounter any issues, feel free to contact us at the email provided in the paper, open issues or even pull-requests directly here.
