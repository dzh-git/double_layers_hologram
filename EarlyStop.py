import numpy as np
import torch
import os
import parameters

class EarlyStop:
    def __init__(self) :
        args=parameters.my_parameters().get_hyperparameter()
        self.save_path=args.model_save_path
        self.patience = 100
        self.best_score=None
        self.counter=0
        self.early_stop = False
        
    
    def __call__(self,accuracy,model):
        score=accuracy
        path=os.path.join(self.save_path,'last.pth')
        if self.best_score is None:
            self.best_score=score
            torch.save(model.state_dict(),path)
        elif score < self.best_score:
            self.counter+=1
            if self.counter >= self.patience:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                self.early_stop = True
        else :
            self.best_score = score
            path=os.path.join(self.save_path,'best.pth')
            torch.save(model.state_dict(),path)
            self.counter = 0
        