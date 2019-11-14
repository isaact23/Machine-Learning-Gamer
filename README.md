# Machine-Learning-Gamer
Program learns how to interact with a game based on video output.

This program interacts with any game that can be emulated on PC. The user begins by 'training' the program by playing the game, and the program develops a neural network, relating the video output of the game window to the keys to press.

The neural network is based off of the one developed by Tariq Rashid in his book "Make Your Own Neural Network". The repo for this is at https://github.com/makeyourownneuralnetwork/makeyourownneuralnetwork.

# How to use
1. Obtain an emulator
2. Configure settings in main.py to point to the emulator window.
  a. Set self.WINDOW_NAME to the beginning of the window name.
  b. Set the PADDING constants to narrow in on the game view.
  c. Adjust self.COMPRESSION based on how many input nodes you want.
  d. Adjust self.LEARNING_RATE and self.KEYPRESS_THRESHOLD as desired.
3. Run main.py. From here a loop will run with options for training and testing.
  T - Train. Click on the game window and move around. Press 1 to exit.
  P - Play. The network will mimic the training and play the game.
  S - Screenshot. Saves a shot.png file with the game window.
  V - Input Matrix. Converts the screenshot to a visualized neural network input.
  E - Exit.

The following link is an example of how the neural network interprets the game. The video stream is converted to monochrome and highly compressed in order to optimize the neural network. This is the input for the neural network; the output is a column vector representing the keys to press.
 
[[https://github.com/shufflesninja/Machine-Learning-Gamer/blob/master/ChainChomp.png|alt=ChainChomp]]
