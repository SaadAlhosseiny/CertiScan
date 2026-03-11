import numpy as np

def apply_mask(image, mask):
  
    return np.multiply(image, mask)

def calculate_score(masked_image):
    
    return np.mean(masked_image)


