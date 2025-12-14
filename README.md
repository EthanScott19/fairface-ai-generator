# FairFace Generator

Fairness aware face generative model using AI-Face

## Overview
This project is based off of the CVPR paper titled **"AI-Face: A Million-Scale Demographically Annotated AI-Generated Face Dataset
and Fairness Benchmark"**. The focus in this project is generation rather than detection.

The original paper studies how well different detectors are able to detect faces across different demographics (race, age, gender, etc.) In this project we are using the AI-Face dataset to generate faces and will be borrowing their metrics to evaluate how balanced the outputs are

**Our  question:**
>Does training a generative model on a demographically annotated, diverse dataset lead to more balanced and fair face generation across different demographics?

# Setup Instructions for Delta

## Dataset
Install dataset by following the links on the [original author's repository](https://github.com/Purdue-M2/AI-Face-FairnessBench/tree/main) in their readme. Make a directory in the root directory of your delta server and copy the dataset there. Example:
```bash
root/
├── dataset/
│   ├── gan_subset1/ #directorys from author's provided dataset
│   ├── gan_subset2/
│   └── gan_subset3
└── projects

```
## Generator
Now in another directory outside of "dataset" clone our repository with 
```bash
git clone https://github.com/EthanScott19/fairface-ai-generator.git
```
### Training
Once you have cloned our repository cd back to the root directory and run the following commands
```bash
salloc --account=bfes-dtai-gh --partition=ghx4 --gpus=1 --time=02:00:00 #change the time to however long needed
ssh ghx    #replace x with whatever gpu you are allocated
module load python/3.10
source my_venv1/bin/activate
cd projects/fairface-ai-generator # or wherever you have cloned our repository

python -m src.train.train_gan \
  --data-root /u/user/dataset \ #change this to where you have downloaded the dataset
  --out-dir outputs/dcgan64 \
  --image-size 64 \
  --latent-dim 128 \
  --batch-size 64 \
  --num-epochs 10 \
  --lr 0.0002 \
  --num-workers 1 \ # you may change the other options but make sure if you are on delta to keep this at 1 
  --device cuda
```
You should now be training the model. To ensure success check out the outputs/dcgan64 directory and see if you are getting checkpoints and samples. If you are, great!

### Generating Samples
Once you have trained for the desired number of epochs, cd back to the project root and run the following:

```bash
python -m src.generate.generate_samples \
  --checkpoint outputs/dcgan64/checkpoints/dcgan_epoch10.pt \ #Change this baased on how many epochs you trained. (This assumes 10)
  --num-samples 2000 \ #Change to number of desired samples
  --batch-size 64 \
  --out-dir generated/dcgan_gan_subset
```
Check generated/dcgan_gan_subset to ensure success. Congratulations, you have successfully generated faces.
# Data comparison (Optional)
**Warning: To do this part you must obtain test_data.csv and train_data.csv from the authors of the paper. We are not permitted to provide these files.**

First, from the project root navigate to the directory labeled "external" and clone the author's repository there. There is a readme file with instructions how to do that located in external. Once you have cloned the authors repository inside external, navigate to their project root and make a directory called "metadata". Within this directory, place the obtained "train_data.csv" and "test_data.csv" files. 

Next navigate to fairface-ai-generator/src/data and modify the "make_gan_csv.py" file so that the columns on the right match the folder names in your dataset and the columns in the left match keywords in the csv files that we just placed in the metadata directory. Then from the project root, run:
```bash
python -m src.data.make_gan_csv
```
Our goal with this script is to trim the annotations so that it matches only the data we used to train our model. If you used the entire dataset then this step is unnecessary. Once you do this you should have a file in fairface-ai-generator/data called "train_gan_subset.csv" which is the trimmed down version. To ensure this worked, from the project root run:
```bash
python -m src.data.test_gan_loader
```
If the output looks like:
```bash
Dataset size: 100000 # or whatever number you are somewhat expecting

Batch image tensor shape: torch.Size([8, 3, 256, 256])
Example label: tensor(1)
Intersection label example: tensor(1)
```
then you are in good shape.

From the project root run:
```bash
python -m src.analysis.analyze_gan_subset
```
This outputs a CSV file breaking down the train_gan_subset.csv file into percentages.


Navigate back to the project root and run this command:
```bash
python -m src.analysis.annotate_generated_clip \
  --input-dir generated/dcgan_gan_subset \
  --out-csv generated/dcgan_gan_subset_clip_labels.csv \
  --device cuda
```
This uses clip to annotate your generated faces with the same rubric used in the author's paper.

Back to the project root, run this command:
```bash
python -m src.analysis.analyze_generated_demographics
```
Similarly to analyze_gan_subset.py, this outputs a CSV breaking down dcgan_gan_subset_clip_labels.csv from the annotated generated faces into percentages.

Once you have all these CSV files, from the project root run:
```bash
python -m src.analysis.plot_skin_tone_comparison
```
This will plot the results in a double bar chart, showing the skin tone distribution between your training set and your actual generated faces. You can modify this value to include gender as well.
