from conx import Network, SRN

class DynamicInputsNetwork(Network):
    def initialize_inputs(self):
        pass
        # Do some initialization here
        # Shuffle them, if self.inputs and "shullfe" is set:
        # if self.settings["shuffle"]:
        #    self.shuffle_inputs()

    def inputs_size(self):
        # Return the number of inputs:
        return 4

    def get_inputs(self, i):
        # Return a pattern:
        temp = [[0, 0], 
                [0, 1], 
                [1, 0], 
                [1, 1]]
        return temp[i]

net = DynamicInputsNetwork(2, 3, 1)

def target_function(inputs):
    return [int(bool(inputs[0]) != bool(inputs[1]))]

net.target_function = target_function

net.train(max_training_epochs=5000,
          tolerance=0.3,
          epsilon=0.1,
          shuffle=True)
net.test()

####

class DynamicInputsSRN(SRN):
    def initialize_inputs(self):
        pass
        # Do some initialization here
        # Shuffle them, if self.inputs and "shullfe" is set:
        # if self.settings["shuffle"]:
        #    self.shuffle_inputs()

    def inputs_size(self):
        # Return the number of inputs:
        return 8

    def get_inputs(self, i):
        # Return a pattern:
        temp = [0, 0, 
                0, 1, 
                1, 0, 
                1, 1]
        return [temp[i]]

net = DynamicInputsSRN(1, 5, 1)

last = 0
def target_function(inputs):
    global last
    retval = [int(bool(inputs[0]) != bool(last))]
    last = inputs[0]
    return retval

net.target_function = target_function

net.train(max_training_epochs=5000,
          tolerance=0.3,
          epsilon=0.1,
          shuffle=True)
net.test()

### 

text = ("This is a test. Ok. What comes next? Depends? Yes. " +
        "This is also a way of testing prediction. Ok. " +
        "This is fine. Need lots of data. Ok?")

letters = list(set([letter for letter in text]))

# Set to zero:
pattern_size = 0

def encode(letter):
    index = letters.index(letter)
    binary = [int(char) for char in bin(index)[2:]]
    if pattern_size:
        return ([0 for i in range(pattern_size)] + binary)[-pattern_size:]
    else:
        return binary

# Reset to max length:
pattern_size = len(encode(letters[-1]))

patterns = {letter: encode(letter) for letter in letters}

class Predict(SRN):
    def initialize_inputs(self):
        pass

    def inputs_size(self):
        # Return the number of inputs:
        return len(text)

    def get_inputs(self, i):
        letter = text[i]
        return patterns[letter]

net = Predict(len(encode("T")), 5, len(encode("T")))

def target_function(inputs):
    index = net.current_input_index
    if index + 1 < len(text):
        letter = text[index + 1]
    else:
        letter = " "
    return patterns[letter]

net.target_function = target_function

net.train(max_training_epochs=5000,
          report_rate=100,
          tolerance=0.3,
          epsilon=0.1,
          shuffle=True)
net.test()
