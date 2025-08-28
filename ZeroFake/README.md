### New
Link to research paper: https://publications.cispa.de/articles/conference_contribution/ZeroFake_Zero-Shot_Detection_of_Fake_Images_Generated_and_Edited_by_Text-to-Image_Generation_Models/27134142?file=49502790

New modifications were taken from: https://github.com/MarkBridge11/ZeroFake-Mod.

In the original code, a hand-crafted list of nouns for the adversarial prompt, along with the cosine similarity calculation for selecting an adversarial prompt were not present. The modfications for these were taken from the above repo.

### Environment

You first need to build the environment by:
```
apt update && apt install -y libsm6 libxext6
conda env create -f env.yaml
conda activate zerofake
```

You also need to download the spacy model by:

```
python -m spacy download en_core_web_sm
```

### Reconstruction

You can reconstruct the given image by:

```
python uni-ddim-inversion.py --target image-path --output output-path
```

The you can compute the similarity between the origianl images and the reconstructed images by:

```
python sim.py --orginal image-path1 --reconstruct image-path2 
```

