# CAST: Counterfactual Labels Improve Instruction Following in Vision-Language-Action Models

[Catherine Glossop](https://catglossop.github.io/),  [William Chen](https://x.com/verityw_), [Arjun Bhorkar](https://scholar.google.com/citations?user=AGPooMYAAAAJ&hl=en), [Dhruv Shah](https://robodhruv.github.io/), [Sergey Levine](https://people.eecs.berkeley.edu/~svlevine/)

## [Visit our website!](https://cast-vla.github.io)
## [ArXiV Link](TODO)


The data augmentation code will be released soon! The training code is available in the `cast-vla` repo linked in this repo, the [CAST dataset](https://huggingface.co/datasets/catglossop/CAST-dataset) and an existing [checkpoint](https://huggingface.co/catglossop/CounterfactualVLA) are available on HuggingFace. 

To get started with training your own model with the CAST dataset, download and unzip the CAST dataset. Follow the instructions in the `cast-vla` repo to start your own training run. 

To start using CounterfactualVLA, download the checkpoint and follow the instructions for inference in the `cast-vla` repo.

Once the inference server has been launched using `cast-vla`, you can launch the robot side client script

```
cd deployment
./navigate_vla.sh '-s <ngrok server address> -w <number of steps to use in generated action chunk> --prompt "your prompt"'
```



