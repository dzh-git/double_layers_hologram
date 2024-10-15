import argparse

class my_parameters:
    def __init__(self):
        super(my_parameters,self).__init__()
        #超参数
        parser=argparse.ArgumentParser()
        parser.add_argument("--img-size",type=int,default=1000)
        parser.add_argument("--batch-size",type=int,default=32)
        parser.add_argument("--num-epochs",type=int,default=2000)
        parser.add_argument('--lr', type=float, default=1e-1, help='学习率')
        
        #model
        parser.add_argument("--model-save-path",type=str,default='./saved_model/') 
        parser.add_argument("--model-load-dir",type=str,default='./saved_model/best.pth') #'./saved_model/best.pth'
        parser.add_argument("--result-record-path",type=str,default='./result.csv')
        self.hyper_parameter=parser.parse_args()
        
        #现实中可能出现的情况：非垂直入射，衍射层错位，衍射层制造存在误差等
        actual_parser=argparse.ArgumentParser()
        actual_parser.add_argument("--distance",type=float,default=0.2) 
        actual_parser.add_argument("--wave-length",type=float,default=266e-9) 
        actual_parser.add_argument("--screen-length",type=float,default=3e-3) 
        actual_parser.add_argument("--oblique-incidence",type=bool,default=False) 
        actual_parser.add_argument("--manufacturing-error",type=bool,default=False) 
        self.actualparameter=actual_parser.parse_args()
        
    
    def get_hyperparameter(self):
        return self.hyper_parameter
    
    def get_actualparameter(self):
        return self.actualparameter
    
