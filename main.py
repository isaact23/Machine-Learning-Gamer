# SUPER MARIO 64 playing algorithm by Isaac Thompson

import neuralnetwork_4_layer as neuralnetwork # Custom module for machine learning
import numpy as np # For matrix creation
import matplotlib.pyplot as plt # Visualize neural network input

import win32gui # Get dimensions of game window
import pyautogui # Read window gui
import keyboard # Detect key presses
import directkeys # Custom module - allows python to send commands to game window

import sys
from time import sleep

KEY_LIST = ['w', 'a', 's', 'd', 'up', 'left', 'down', 'right']

# The object that plays games
class N64_AI():

    # Game-playing object creation
    def __init__(self):

        # Change this to whatever your emulator window name is
        self.WINDOW_NAME = WINDOW_NAME = "Super Mario 64 (U)"

        # Find window dimensions
        #self.DIMENSIONS = self.get_window_dimensions()
        self.DIMENSIONS = (0, 80, 640, 450)
        assert self.DIMENSIONS != None, "Error: Could not find game window " + WINDOW_NAME
        #LEFT   = self.DIMENSIONS[0]
        #TOP    = self.DIMENSIONS[1]
        WIDTH  = self.DIMENSIONS[2]
        HEIGHT = self.DIMENSIONS[3]

        # Neural network optimization.
        # Use 1 for no compression. Higher numbers will reduce the size of the neural network and input nodes.
        self.COMPRESSION = 8

        # Determine, based on compression, the width and height of the compressed input image.
        self.COMPRESSED_WIDTH = round(WIDTH   / self.COMPRESSION)
        self.COMPRESSED_HEIGHT = round(HEIGHT / self.COMPRESSION)

        # Determine the size of each layer.
        self.INPUT_NODES = self.COMPRESSED_WIDTH * self.COMPRESSED_HEIGHT
        self.OUTPUT_NODES = 8
        self.HIDDEN_NODES_1 = round((self.INPUT_NODES * 2 + self.OUTPUT_NODES) / 3)
        self.HIDDEN_NODES_2 = round((self.INPUT_NODES + self.OUTPUT_NODES * 2) / 3)

        # Neural network settings
        self.LEARNING_RATE = 0.001 # How quickly nodes will be adjusted to compensate for error.
        self.KEYPRESS_THRESHOLD = 0.5 # The neural network must output a value higher than this in order for it to be considered a press.

        # Instantiate a neural network object
        self.neural_network = neuralnetwork.neuralNetwork( inputnodes   = self.INPUT_NODES,
                                                           hiddennodes1 = self.HIDDEN_NODES_1,
                                                           hiddennodes2 = self.HIDDEN_NODES_2,
                                                           outputnodes  = self.OUTPUT_NODES,
                                                           learningrate = self.LEARNING_RATE )

    # Get dimensions of the game window
    def get_window_dimensions(self):

        dimensions = []
        def callback(window, dimensions):
            if self.WINDOW_NAME in win32gui.GetWindowText(window):
                dimensions.append(win32gui.GetWindowRect(window))
            
        win32gui.EnumWindows(callback, dimensions)
        if dimensions: return dimensions[0]
        return None

    # Get full screenshot of the game window
    def get_screenshot(self):

        return pyautogui.screenshot(region=self.DIMENSIONS)

    # Show the game window screenshot in full resolution.
    def show_screenshot(self):
        
        pyautogui.screenshot('shot.png', region=self.DIMENSIONS)
        print('Screenshot saved to shot.png')

    # Create a matrix of video data from the game (neural network input)
    def get_screenshot_matrix(self):
        
        # Take a box screenshot of the SM64 window
        img = self.get_screenshot()

        # Convert screenshot to a raw array        
        img_array = np.array( img, dtype="int32" )

        # Convert the array to a reduced, grayscale one-dimensional array
        new_img_array = np.zeros(( self.COMPRESSED_HEIGHT * self.COMPRESSED_WIDTH ))
        for y in range( self.COMPRESSED_HEIGHT ):
            for x in range( self.COMPRESSED_WIDTH ):
                nx = round(x * self.COMPRESSION + (self.COMPRESSION / 2))
                ny = round(y * self.COMPRESSION + (self.COMPRESSION / 2))
                new_img_array[x * self.COMPRESSED_HEIGHT + y] = img_array[ny][nx][0] + img_array[ny][nx][1] + img_array[ny][nx][2]

        return new_img_array

    # Show the user what the neural network sees.
    def show_screenshot_matrix(self):
        
        img = self.get_screenshot_matrix()
        img = np.reshape( img, ( self.COMPRESSED_HEIGHT, self.COMPRESSED_WIDTH ), order='F')
        plt.imshow( img, cmap='Greys', interpolation='None' )
        plt.show()

    # Create a matrix of keyboard data from the user (neural network output)
    def get_keyboard_data(self):
        
        output = np.zeros(8)
        for i, key in enumerate(KEY_LIST):
            if keyboard.is_pressed(key):
                output[i] = 0.99
            else:
                output[i] = 0.01
                
        return output

    # Train the neural network with input and target data (video and keyboard)
    def train(self):
        
        video_data = self.get_screenshot_matrix()
        keyboard_data = self.get_keyboard_data()
        #print(np.round_(keyboard_data, decimals=2))
        self.neural_network.train(video_data, keyboard_data)

    # Feed video data into the neural network to obtain keys to press.
    def query(self):
        
        movement = self.neural_network.query(self.get_screenshot_matrix())
        print(np.round_(movement, decimals=2))
        return movement

    # Learn from the player's movements until escape is pressed.
    def learn_from_player(self):
        
        sleep(1)
        print("Learning. Press 1 to stop.")
        while not keyboard.is_pressed('1'):
            AI.train()
        print("Done learning.")

    # Use what we know to play the game.
    def play(self):
        
        sleep(1)
        print("Playing. Press 1 to stop.")
        while not keyboard.is_pressed('1'):
            movement = AI.query()
            for i, key in enumerate(KEY_LIST):
                if movement[i] > self.KEYPRESS_THRESHOLD:
                    directkeys.PressKey(key)
                else:
                    directkeys.ReleaseKey(key)
        print("Done playing.")
        for key in KEY_LIST:
            directkeys.ReleaseKey(key)

    # Test python's ability to create keystrokes that affect the emulation.
    def test_movement(self):

        print("Testing movement now.")
        
        directkeys.PressKey('up')
        sleep(5)
        directkeys.ReleaseKey('up')
        
        print("Done testing movement.")

    # Main function allowing user to control training, playing, and stage creation.
    def user_control(self):
        
        print("Welcome to the Game Learner by Isaac Thompson.")
        print("Window dimensions are " + str(self.DIMENSIONS))
        self.show_screenshot()
        while True:
            action = 'none'
            while not action in 'tepsv':
                action = input("Type T to train, P to play, E to exit, S for screenshot, V to see input ").lower()
            if action == 'e':
                sys.exit()
            elif action == 't':
                self.learn_from_player()
            elif action == 'p':
                self.play()
            elif action == 's':
                self.show_screenshot()
            elif action == 'v':
                self.show_screenshot_matrix()

print("Loading...")
AI = N64_AI()
AI.user_control()
