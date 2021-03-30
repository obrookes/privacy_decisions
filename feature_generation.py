import torch
from torch.nn import functional as F

class FeatureGeneration:
    def __init__(self,model,classes,img_ids):
        self.model=model
        self.classes=classes
        self.img_ids=img_ids
        self.outputs=None
        self.results=[]

    def show_model(self):
        print(self.model)

    def get_classes(self):
        return self.classes

    def run_model(self,image_batch):
        self.model.eval()
        with torch.no_grad():
            outputs=self.model(image_batch)
            self.outputs=outputs

    def evaluate(self,num_results):
        for output,img_id in zip(self.outputs,self.img_ids):
            r={}
            prob=F.softmax(output,dim=0)
            top_p,top_c=torch.topk(prob,num_results)
            try:
                r['id']=img_id.strip('.jpg')
            except:
                r['id']=img_id.strip('.png')
            r['prob']=top_p.numpy()
            r['class']=list(map(self.classes.__getitem__,top_c.numpy()))
            self.results.append(r)

    def get_results(self):
        return self.results

    
            
    

