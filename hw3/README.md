# Homework 3
In this last homework, you will leverage a trained diffusion model to produce images that meet various requirements.

A base code is provided here: https://colab.research.google.com/drive/1VICxaD-RZFWCvytk5aWQrmdDzrnJskY1
You can run it online on Google Colab or on your machine. This code loads and defines everything necessary to generate an image conditioned on a text prompt using classifier-free guidance. Your goal will be to adapt the sampling process. 

## Tasks
### Warm-Up
 * Generate an image of your choice using the provided code.
 * Vary the parameter $N$, what do you observe? Does it agree with your expectations?

### Interpolation
 * Implement a new version of the `sample` function that receives a list of prompts and associated real numbers (weights) in inputs. Each generation iteration should follow a weighted average of the `eps` computed for each prompt. In the provided code this is done in the computation `eps_uncond*(-6.5) + eps_cond*7.5`, you must thus generalize this averaging to multiple prompts.
 * Run your code with three prompts and weights of your choice. Note that the weights should sum to one. Example: `prompts= ['low resolution', 'House', 'In the woods',], weights =[-6, 4.5, 2.5]`. Your produced image should illustrate the satisfaction of the positively weighted prompts and the repelling of a negatively weighted one.
 * Implement and execute a function leveraging your new sample function to generate a sequence of images slowly morphing from one type of image into another. For this, gradually diminish the weight given to one prompt and increase the weight given to another (while keeping a fixed random seed).

### Illusion
You will implement the idea described in this paper: https://arxiv.org/pdf/2311.17919
The idea is to average several directions in each generation step to generate an image following several prompts under different views.
  * Implement a new version of the sample function that implements this idea. Here we use the empty prompt `""` and two other prompts. Implement the following update pseudocode at each generation iteration:
    - `eps00 = model(xt, "")`
    - `eps01 = model(xt, prompt1)`
    - `eps10 = model(flip(xt), "")`
    - `eps11 = model(flip(xt), prompt2)`
    - `eps = (1-c)/2*(eps00+flip(eps10)) + c/2*eps01 + c/2*flip(eps11)`
    - `x_{t-1} = xt - (sig_t - sig_{t-1}) eps`
  * Generate two illusions.
  * Propose and implement your own way of creating illusions based on this idea.

## Guidelines
Submit your codes and report in a single zip file on Moodle by Friday 20 December 23h59. Do not forget to register for a group again for this third homework.
Your report should be concise and clear, in pdf format, and with maximum 10 pages (all included). Make sure you use pdf fromat for your images as this will make them easier to read. Do not forget to include a few words about the methods that you use to show the reader that you understand what you are actually doing.

Best of luck!
