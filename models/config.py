
import os
import pandas as pd
import torch 
from pathlib import Path



config_dictionary_1D = dict(random_state=1001,
                            num_classes=2,
                            lr= 1e-4,
                            weight_decay= 1e-4, 
                            dropout= 0.3,
                            nhead= 8, 
                            dim_feedforward= 512,
                            batch_size= 256,
                            early_stop_tolerance= 10,
                            epochs= 50, 
                            )


                                            
                
                
                
                
