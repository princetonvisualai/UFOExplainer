# UFOExplainer
Code for UFO: A unified method for controlling Understandability and Faithfulness Objectives in concept-based explanations for CNNs

To cite, 
```
@misc{ramaswamy2023ufo,
      title={UFO: A unified method for controlling Understandability and Faithfulness Objectives in concept-based explanations for CNNs}, 
      author={Vikram V. Ramaswamy and Sunnie S. Y. Kim and Ruth Fong and Olga Russakovsky},
      year={2023},
      eprint={2303.15632},
      archivePrefix={arXiv},
}
```

Steps to run

1. Download Broden images + annotations as in https://github.com/CSAILVision/NetDissect-Lite 
2. Get scene scores as done in https://github.com/princetonvisualai/OverlookedFactors   
3. Use `load_data.py` to preprocess data
4. Use `attr_rot.py` to learn attribute classifiers
5. Use `learn_attr_cutoff.py` to learn explanations. 
6. `analsis.ipynb` contains the analysis done in the paper. 
