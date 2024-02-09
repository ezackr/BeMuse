# BeMuse
A tool that is able to match a given MusicXML query to the most similar piece in a database of classical music.

## Setup

### Melody checkpoint

Follow the instructions on the [MidiBERT](https://github.com/wazenmai/MIDI-BERT/tree/CP) GitHub page to download the MidiBERT model checkpoints. Save the `pretrain_model.ckpt` file in the `BeMuse/artifact/midibert` directory.

## Overview

We provide a brief overview of the project setup and the steps used to analyze a MIDI input query. For clarity, a diagram of the model architecture is provided below:

![BeMuse architecture diagram](./media/bemuse-architecture.png)

BeMuse begins by preprocessing a MIDI input file into a compound word sequence (CP), which is passed as input to a fine-tuned MidiBERT encoder. 
The encoder outputs a 768-dimensional vector representation of the input in latent space. 
Then, the input is compared against a pre-computed database of song vectors using cosine similarity. 
After comparing the input vector against all possible outputs, the tool outputs the top 5 most-similar songs to the given input.

