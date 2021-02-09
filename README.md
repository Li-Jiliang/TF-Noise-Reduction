# TF-Noise-Reduction
A Residual Deep Bidrectional LSTM Netowrk written in Tensorflow

## Requirements

1) Tensorflow 2.1 (The offical docker image is recommended)
8) CUDA
9) CUDNNLSTM
2) Keras
3) Numpy
4) Librosa
5) Noise Reduce 
6) Scipy
7) SKLearn

Eveything else that is required to trian the neural network is contained within the Jupyter Notebook, the only exception being the training datasets (links below). Once downloaded, replace the paths in the Jupyter Notebook with your local paths to the folders. If training inside of the official Tensorflow docker image (recommended) bind mount the datasets directories to '~/docker_ws/Datasets/<dataset_name>'

## System Overview
![Ststem Overview](/images/System_Overview.png)

1) One Clean voice track and one randomly selected noise track are loaded from their respective datasets

2) The volume of the noise is reduced by a random value between 0 and 75 percent

3) The noise sample is mixed with the clean sample, the resulting noisy mixture serves as the input to the neural network while the (unaltered) clean sample serves as it’s output target.

4) A STFT is performed on both samples.

5) The noisy mixture signal is fed to the neural network as the input
while the original clean sample used as the output target.

6) The current noise sample is removed from the list of available
noise samples to ensure the model is trained on all noise samples in the dataset.

7) Steps 2-6 are repeated for each clean audio sample in the clean
dataset and until they run out (unlikely) or the eatly stopping criteria is met.


## Layers and Connections
![Model Layers](/images/layers.png)


## Datasets   

In order to reduce the ram required to train this model, some of the training data is generated on the fly, while the rest is loaded from disk these behavours can be controlled by modifing the generator class in the Jupyter Notebook

1) [Mozilla Common Voice (MCV)](https://commonvoice.mozilla.org/en)
MCV is an open-source dataset comprised of (1,488 hours) of English sentences read by 51,072 separate individuals of varying genders / ethnic backgrounds. The data is provided as 230,000 mp3 files, for the purposes of this project a random selection of 10,000 of these recordings have been converted to wav format for use in training, validation and testing of the neural network.
Since these files were intended for use in voice recognition and were submitted by untrained volunteers, many of the recordings are quite noisy, therefore all samples used in the training, testing and validation of the neural network have had noise reduction applied, using the method described in x. The MCV dataset was split into an 80,10,10 train, validation, test respectively.  

2) [MIR for Mirex (MIR)](https://www.music-ir.org/mirex/wiki/MIREX_HOME)  
The MIR dataset is designed for the research of singing voice separation, it is comprised of 1000 song clips ranging from 4 to 13 seconds. The clean voice samples from this dataset are mixed with the noise files and used in training, validation and testing of the neural network.  

3) [UrbanSound 8K (U8K))[https://urbansounddataset.weebly.com/urbansound8k.html]
The U8K dataset contains 8732 audio files of urban sounds in WAV format. The sampling rate, bit depth, and number of channels vary from file to file, all samples used in the training of the model were re sampled to 16bit 16000Hz mono.  



## Feature Design

![Dataset Structure](/images/LSTM_Dataset.png)

As previously mentioned the samples from the MCV dataset have noise reduction applied to them in the pre-processing stage in order to make them into an acceptable target for the neural network.

Before the Short Time Four Transfer (STFT) is computed to create the input and output targets for the neural network, each sample is normalized to the (-1,1) range, studies have shown this to improve the speed of training neural networks as well as helping to avoid dramatic shifts in the weights of a neuron that they may not recover from - rendering it useless from that point on wards.

The LSTM network expects the input data (X) to be provided with a specific array structure in the form of: [Batch Size, Time Steps, Features]. In the case of this project Batch size = an individual voice recording, Time Steps: the time step at which each feature sample was taken, Features: the magnitude of the Fourier transformed windowed sample.  

Ideally each individual sample would be of equal length and if not they can be ’padded’ with zeros so that each sample is the exact length of the longest sample in the dataset, a masking layer can then be added to the network so that these additional zeros are ignored during training, major drawback of this approach is the massively increases training time for little gain since in both the MCV and MIR datasets the shortest sample was more than 20x smaller than the longest.

Another approach would be to truancate all sequences that are longer than the shortest sequence in the dataset to the length of the shortest sequence, this however in all three datasets this approach would end up discarding a majority of the dataset. Models trained with this approach perform, poorly as expected.

The current approach is a combination of the padding and truncation approaches, each sample longer than the shortest sample in the dataset is sliced into sequences equal to that of the shortest sequence with any remainder becoming a sample of its own. Any of the shorter samples are then zero-padded to make them equal to the length of the shortest sequence.

This approach keeps the benefits of both of the previous approaches whilst having none of the drawbacks. This is the approach used in the jupyter notebook.

Finally, taking the above into account input to the neural network is as below:

- Batch size is 1
- Time Steps are the time axis of the STFT
- Features are the Magnitudes at each STFT time step.

With the STFT of the signals being computed with the following parameters:  

• Window Length: 512  
• Overlap: 128  
• FFT Length: 512  

## Phase Recovery

Since neural networks implemented TensorFlow cannot operate on complex-valued data, only the magnitude data from STFTs computed from  the samples in the dataset are fed to the neural network. This means that the phase information must somehow be restored to the denoised audio output of the neural network so that useful audio can be obtained.

Two approaches are explored:

1) **Griffin-Lim Algorithm:

The Giffin-Lim approximation is a method that can be used to estimate the phase of a time-series waveform using only its magnitude components.
Griffin-Lim is able to converge towards the estimated phase spectrum of a given amplitude-only waveform through the application of the following steps

1) A complex matrix is created, equal in size to the size of the matrix containing the magnitude values of a STFT waveform.
2) An imaginary matrix of uniform noise is created in and inserted into the place of the missing phase information
3) an iSTFT performed on the matrix, at this point we now have a time domain signal based only on the magnitude data from the original signal
4) an STFT is performed on the time series, this extracts a small amount of useful phase information however the signal is still far from acceptable at this point
5) In this new spectogram the amplitude information is replaced with the amplitude information from the original matrix, at this point we have a matrix containing the original amplitude data and a small amount of extracted phase information.
6) Steps 2-5 are repeated, since the Grffin-Lim algorithm is conver- gent, the phase information becomes more and more useful with each iteration.
One important factor to note is that the final iteration output of the Griffin-Limed signal may be slightly shorter/longer than the original data. This is due to the resolution rounding that is performed in the STFT process.

2) **Noisy Signal Phase: 

 An alternative to the Griffin-Lim approach is to re-introduce the phase information from the original noisy waveform into the output from the neural network, this approach is faster than the Griffin-Lim approach as it only requires a single operation however it can potentially re-introduce some of the noise from the original signal.
 
 Both of the aformentioned approaches have been implemented in the Jupyter Notebook.


## Example Outputs

### Clean audio input
![Model Layers](/images/cleanAudio.png)
### Clean audio input + Noise
![Model Layers](/images/noiseAdded.png)
### Denoised signal from neural network
![Model Layers](/images/magOnlyPrediction.png)
### Denoised signal from neural network + phase from original (noisy) signal
![Model Layers](/images/phaseRestoredPrediction.png)
### Denoised signal from neural network + phase computed using Griffin-Lim approximation
![Model Layers](/images/6Layers_griffinlim.png)

### Comparisons
### Original Singal vs NN Ouput + phase from original (noisy signal)
![Model Layers](/images/comparison.png)
### Original Singal vs NN Ouput + phase computed using Griffin-Lim approximation
![Model Layers](/images/comparison1.png)
