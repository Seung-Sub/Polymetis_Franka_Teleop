"""
To collect input data for naive quantization's calibration data 
"""

# collect input data for calibration data 

global diffusion_input_list
diffusion_input_list = []

def appendInput(value):
    diffusion_input_list.append(value)

def getInputList():
    return diffusion_input_list 

def resetInput():
    diffusion_input_list.clear()