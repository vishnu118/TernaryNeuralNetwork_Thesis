#!/usr/bin/env python3
#SBATCH --partition=elec.gpu.q
import sys
import os
sys.path.append(os.getcwd()) 
import comet_ml
import os
import random
import pl_bolts
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics
import torchvision
import argparse
import configparser
from pytorch_lightning.utilities.cli import LightningCLI
from pytorch_lightning.utilities.seed import seed_everything
from torch.utils.data import DataLoader
from torch import nn
from multiprocessing import process
import Quant_Collector
from Quant_Collector import Update_Epoch_Counter
from Quant_Collector import STE_FWD, STE_BWD
from Quant_Collector import TWN_Quant, TTQ_Quant, FGQ_Quant
from Quant_Collector import ESA_Quant, ESA_Loss_addition
from Quant_Collector import round_ste, Load_matching_dicts
from Quant_Collector import FATNN_Act_Quantizer, FATNN__Weight_Quantizer
from Quant_Collector import RPreLU, RSign_Ternary
from torchvision.datasets import CIFAR100
import torchvision.transforms as tt
CUDA_LAUNCH_BLOCKING=1
#global_logger = None
#dataset_type = "cifar10"
dataset_type = "cifar100"


#Basic_Block_type = "Normal"
#Basic_Block_type = "Bottleneck"
Basic_Block_type = "MBConv"


name = "Placeholder"
seed = 1984
lr = 0.1
lr_scheduling = "Cosine"
LoadPretrained = False

###########  Global Weight Quantisation stuff  ###########

Weight_quant_forward = "ESA"
Weight_quant_backward = "STE"

###########  Global Act. Quantisation stuff  ###########

Act_quant_forward = "FATNN"
Act_quant_backward = "STE"

###########  EWGS stuff  ###########

EWGS_Weights = False
EWGS_Factor_Weights = 10
EWGS_Act = False
EWGS_Factor_Act = 10

######### Arch. changes ############

RPrelu = False
Extra_Skip = False
Two_Step = False
Phase = 1 




def load_config_file_parameters(Config_parser):
    # [Experiment] section 
    global seed
    global lr
    global lr_scheduling
    global LoadPretrained
    global name
    global dataset_type
    Experiment = "Experiment"
    name = Config_parser[Experiment].get("name")
    seed = Config_parser[Experiment].get("seed")
    lr = Config_parser[Experiment].get("lr")
    lr_scheduling = Config_parser[Experiment].get("lr_scheduling")
    LoadPretrained = Config_parser[Experiment].get("LoadPretrained")


    # [Weight Quantization] section
    global Weight_quant_forward
    Weight_Quantization = "Weight Quantization"
    Weight_quant_forward = Config_parser[Weight_Quantization].get("Weight_quant")

    # [Activation Quantization] section
    global Act_quant_forward
    Activation_Quantization = "Activation Quantization"
    Act_quant_forward = Config_parser[Activation_Quantization].get("Act_quant")


    # [Quantizer Enhancements] section 
    global Weight_quant_backward
    global Act_quant_backward
    global EWGS_Weights 
    global EWGS_Factor_Weights 
    global EWGS_Act
    global EWGS_Factor_Act 


    Quantizer_Enhancements = "Quantizer Enhancements"
    Weight_quant_backward =  Config_parser[Quantizer_Enhancements].get("Weight_Quant_Enhance")   
    Act_quant_backward =  Config_parser[Quantizer_Enhancements].get("Act_Quant_Enhance")   
    EWGS_Weights =  Config_parser[Quantizer_Enhancements].get("isEWGS_Weights")   
    EWGS_Act =  Config_parser[Quantizer_Enhancements].get("isEWGS_Act")   
    EWGS_Factor_Weights =  Config_parser[Quantizer_Enhancements].get("EWGS_Factor_Weights")   
    EWGS_Factor_Act =  Config_parser[Quantizer_Enhancements].get("EWGS_Factor_Act")   



    # [Architectural Changes]
    global RPrelu 
    global Extra_Skip 
    global Two_Step 

    Architectural_Changes = "Architectural Changes"
    RPrelu = Config_parser[Architectural_Changes].get("RPreLU")  
    Extra_Skip = Config_parser[Architectural_Changes].get("Extra_Skip")  
    Two_Step = Config_parser[Architectural_Changes].get("Two_Step")  


    #[Teacher Student apporach]


    print("config parser read")
    return 0

def log_all_hyperparams(logger):
    global seed
    global lr
    global lr_scheduling
    global LoadPretrained
    global Weight_quant_forward
    global Act_quant_forward
    global Weight_quant_backward
    global Act_quant_backward
    global EWGS_Weights 
    global EWGS_Factor_Weights 
    global EWGS_Act
    global EWGS_Factor_Act 
    global RPrelu
    global Extra_Skip 
    global Two_Step 
    global dataset_type
    logger.log_parameter("Seed",seed)
    logger.log_parameter("Learning Rate",lr)
    logger.log_parameter("Dataset Selected is ",dataset_type)
    logger.log_parameter("LR Scheduling method",lr_scheduling)
    logger.log_parameter("Load Pretrained model",LoadPretrained)
    logger.log_parameter("Weight Quant Forward Method",Weight_quant_forward)
    logger.log_parameter("Act Quant Forward Method",Act_quant_forward)
    logger.log_parameter("Weight Quant Backward Method",Weight_quant_backward)
    logger.log_parameter("Act Quant Backward Method",Act_quant_backward)
    logger.log_parameter("EWGS Weights enabled",EWGS_Weights)
    logger.log_parameter("EWGS Act enabled",EWGS_Act)
    logger.log_parameter("EWGS Weights factor",EWGS_Factor_Weights)
    logger.log_parameter("EWGS Act factor",EWGS_Factor_Act)
    logger.log_parameter("RPReLU enabled",RPrelu)
    logger.log_parameter("Extra skip connection enabled",Extra_Skip)
    logger.log_parameter("Two step training enabled",Two_Step)

    return 0



class TernarySTE(nn.Module):
    def __init__(self):
        super(TernarySTE, self).__init__()
        global Act_quant_forward

        
        # Forward Pass method selection 
        if(Act_quant_forward == "FATNN"):
            print("FATNN activation on")
            print("EWGS : " + str(EWGS_Factor_Act))
            self.q_func = FATNN_Act_Quantizer(Act_quant_backward, EWGS_Factor_Act)
        elif(Act_quant_forward == "RSIGN"):
            print("RSign activation on")
            print("EWGS : " + str(EWGS_Factor_Act))
            self.q_func = RSign_Ternary(Act_quant_backward, EWGS_Factor_Act)
        else:
            print("FP activations")
        #    self.q_func = STE_FWD


    def forward(self, input):
        global Act_quant_forward
        global Act_quant_backward

        if(Act_quant_forward == "FP"):
            output = input
        else:
            output = self.q_func(input)


        return output


class Ternarize(nn.Module):

    def __init__(self):
        super(Ternarize, self).__init__()
        global Weight_quant_forward
        global Weight_quant_backward
        
        # Forward Pass method selection 
        if(Weight_quant_forward == "TWN"):
            print("TWN weight on")
            self.q_func = TWN_Quant(Weight_quant_backward, EWGS_Factor_Weights)
        elif(Weight_quant_forward == "TTQ"):
            print("TTQ weight on")
            self.q_func = TTQ_Quant(Weight_quant_backward, EWGS_Factor_Weights)
        elif(Weight_quant_forward == "FGQ"):
            print("FGQ weight on")
            self.q_func = FGQ_Quant(Weight_quant_backward, EWGS_Factor_Weights)
        elif(Weight_quant_forward == "ESA"):
            print("ESA weight on")
            self.q_func = ESA_Quant(Weight_quant_backward, EWGS_Factor_Weights)
        elif(Weight_quant_forward == "FATNN"):
            print("FATNN Weights")
            self.q_func = FATNN__Weight_Quantizer(Weight_quant_backward, EWGS_Factor_Weights)
        else:
            print("FP weight on")
        #    self.q_func = STE_FWD()

    def forward(self, input):
        global Weight_quant_forward
        global Weight_quant_backward

        if(Weight_quant_forward == "FP"):
            output = input
        else:
            output = self.q_func(input)

            

        return output



class TernaryConv2d(nn.Conv2d):
    grouping = 1

    def __init__(self,in_planes, planes, kernel_size, stride, padding, bias,groups=1):
        super(TernaryConv2d,self).__init__(in_channels=in_planes, out_channels=planes, kernel_size=kernel_size, stride=stride, padding=padding, bias = bias, groups = groups)
        
        self.quant_w = Ternarize()
        #print("Groups within this convolution are: " + str(groups))
        self.grouping = groups

    def forward(self, input):

        if((Phase == 1)and(Two_Step == "True")): 
            return F.conv2d(input=input, weight=self.weight, bias=self.bias, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.grouping)
        elif((Phase == 2)and(Two_Step == "True")): 
            return F.conv2d(input=input, weight=self.quant_w(self.weight), bias=self.bias, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.grouping)
        else:
            return F.conv2d(input=input, weight=self.quant_w(self.weight), bias=self.bias, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.grouping)

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)





class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride):
        super(BasicBlock, self).__init__()
        global RPrelu

        
        self.act0 = TernarySTE()
        self.conv1 = TernaryConv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        if(RPrelu == "True"):
            self.relu1 = RPreLU(planes)
        else:
            self.relu1 = nn.ReLU()

        self.act1 = TernarySTE()
        self.conv2 = TernaryConv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        if(RPrelu == "True"):
            self.relu2 = RPreLU(planes)
        else:
            self.relu2 = nn.ReLU()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = LambdaLayer(
                lambda x: F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes // 4, planes // 4), "constant", 0))




    def forward(self, x):
        out = self.act0(x)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.act1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + self.shortcut(x)
        out = self.relu2(out)
        return out



class BasicBlock_extraskip(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride):
        super(BasicBlock_extraskip, self).__init__()
        global RPrelu
        global Extra_Skip
        
        self.act0 = TernarySTE()
        self.conv1 = TernaryConv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        if(RPrelu == "True"):
            self.relu1 = RPreLU(planes)
        else:
            self.relu1 = nn.ReLU()

        self.act1 = TernarySTE()
        self.conv2 = TernaryConv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        if(RPrelu == "True"):
            self.relu2 = RPreLU(planes)
        else:
            self.relu2 = nn.ReLU()

        self.shortcut = nn.Sequential()
        self.shortcut2 = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = LambdaLayer(
                lambda x: F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes // 4, planes // 4), "constant", 0))




    def forward(self, x):
        out = self.act0(x)
        out = self.conv1(out)
        out = self.bn1(out)
        out = out + self.shortcut(x)
        out = self.relu1(out)
        res1 = out
        out = self.act1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + self.shortcut2(res1)
        out = self.relu2(out)
        return out





class BasicBlock_BottleNeck(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride):
        super(BasicBlock_BottleNeck, self).__init__()
        global RPrelu

        ### Pointwise layer already doing channel upsampling but at same size 
        self.act1 = TernarySTE()
        self.conv1 = TernaryConv2d(in_planes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        if(RPrelu == "True"):
            self.relu1 = RPreLU(planes)
        else:
            self.relu1 = nn.ReLU()

        ## Same number of channels but we have size downsampling here 
        self.act2 = TernarySTE()
        self.conv2 = TernaryConv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        if(RPrelu == "True"):
            self.relu2 = RPreLU(planes)
        else:
            self.relu2 = nn.ReLU()


        ## Maintain size and channel stuff 
        self.act3 = TernarySTE()
        self.conv3 = TernaryConv2d(planes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)

        if(RPrelu == "True"):
            self.relu3 = RPreLU(planes)
        else:
            self.relu3 = nn.ReLU()



        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = LambdaLayer(
                lambda x: F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes // 4, planes // 4), "constant", 0))




    def forward(self, x):
        out = self.act1(x)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.act2(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out = self.act3(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out = out + self.shortcut(x)
        out = self.relu3(out)
        return out






class BasicBlock_MB(nn.Module):
    expand_ratio = 6

    ## Basic block of 1 depthwwise and 1 pointwise convolution 
    ## Following each is a batch norm 
    ## Swish non linearities are used

    def __init__(self, in_planes, planes, stride):
        super(BasicBlock_MB, self).__init__()
        global RPrelu

        final_planes = planes
        # Expansion phase (Inverted Bottleneck)
        out_planes = in_planes*self.expand_ratio


        if self.expand_ratio != 1:
            self.expand_conv = TernaryConv2d(in_planes, out_planes, kernel_size=1, stride=1,padding=0, bias=False)
            self.expand_bn = nn.BatchNorm2d(out_planes)



        self.act1 = TernarySTE()
        # Depthwise
        # perform all downsamplig here 
        self.conv1 = TernaryConv2d(out_planes, final_planes, kernel_size=3, stride=stride, padding=1, bias=False, groups = planes)
        self.bn1 = nn.BatchNorm2d(final_planes)

        if(RPrelu == "True"):
            self.relu1 = RPreLU(final_planes)
        else:
            # AS swish is used in the paper
            self.relu1 = nn.SiLU()

        self.act2 = TernarySTE()
        #Pointwise
        self.conv2 = TernaryConv2d(final_planes, final_planes, kernel_size=1, stride=1,padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(final_planes)

        if(RPrelu == "True"):
            self.relu2 = RPreLU(final_planes)
        else:
            # AS swish is used in the paper
            self.relu2 = nn.SiLU()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = LambdaLayer(
                lambda x: F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes // 4, planes // 4), "constant", 0))


    def forward(self, x):
        out = self.act1(x)
        #print("BB input 0 = " + str(out.size()))
        if self.expand_ratio != 1:
            out = self.expand_conv(out)
            out = self.expand_bn(out)
            #print("BB input 1 = " + str(out.size()))
        out = self.conv1(out)
        out = self.bn1(out)
        #print("BB input 2 = " + str(out.size()))
        out = self.relu1(out)
        out = self.act2(out)
        out = self.conv2(out)
        #print("BB input 3 = " + str(out.size()))
        out = self.bn2(out)
        out = out + self.shortcut(x)
        out = self.relu2(out)
        return out

















class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 16
        
        global RPrelu
        
        if(RPrelu == "True"):
            self.stem = nn.Sequential( nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False), nn.BatchNorm2d(16), RPreLU(16))
        else:
            self.stem = nn.Sequential( nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False), nn.BatchNorm2d(16), nn.ReLU())

        self.features = nn.Sequential(
            self._make_layer(block, 16, num_blocks[0], stride=1),
            self._make_layer(block, 32, num_blocks[1], stride=2),
            self._make_layer(block, 64, num_blocks[2], stride=2)
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, num_classes)
        )

        self.apply(self._weights_init)

    def _weights_init(self, m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)
        

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes  # * block.expansion # TODO: expansion disabled for now

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.stem(x)
        #print("Stem output 0 = " + str(out.size()))
        out = self.features(out)
        out = self.classifier(out)
        return out


class BaseModel(pl.LightningModule):
    def __init__(self, warmup_epochs=2, **kwargs):
        super().__init__()

        if(dataset_type == "cifar100"):
            print("CIFAR100 selected")
            num_classes = 100
        else:
            print("CIFAR 10 Selected")
            num_classes = 10

        global Extra_Skip 
        if(Extra_Skip == "True"):
            self.backbone = ResNet(BasicBlock_extraskip, [3, 3, 3],num_classes = num_classes)  # ResNet-20

        if(Basic_Block_type == "Bottleneck"):
            self.backbone = ResNet(BasicBlock_BottleNeck, [3, 3, 3],num_classes = num_classes)
        elif(Basic_Block_type == "MBConv"):
            self.backbone = ResNet(BasicBlock_MB, [3, 3, 3],num_classes = num_classes)
        else:
            self.backbone = ResNet(BasicBlock, [3, 3, 3],num_classes = num_classes)  # ResNet-20

        if(LoadPretrained == "True"):
            checkpoint = torch.load("FP.ckpt")
            self = Load_matching_dicts(checkpoint['state_dict'],self)
            print("Loading of Pretrained model succeeded")

        #global Weight_quant_forward
        #if(Weight_quant_forward == "TTQ" or Weight_quant_forward == "FGQ"):
        #    checkpoint = torch.load("FP.ckpt")
        #    pretrained_state = checkpoint['state_dict']
        #    self.backbone = Load_matching_dicts( pretrained_state, self.backbone)
        if((Phase == 2)and(Two_Step == "True")):
            load_file = name + "_Phase1.ckpt"
            checkpoint = torch.load(load_file)
            self.load_state_dict(checkpoint['state_dict'])  
        
        
        self.loss = nn.CrossEntropyLoss()


        # Metric collections
        self.train_metrics = torchmetrics.MetricCollection([torchmetrics.Accuracy()], prefix='training/')
        self.validation_metrics = self.train_metrics.clone(prefix='validation/')
        self.test_metrics = self.train_metrics.clone(prefix='test/')
        # Save hyperparameters
        self.save_hyperparameters()

    def forward(self, x):
        return self.backbone(x)

    def _step(self, batch, batch_idx, prefix, metric_collection, on_step=None, on_epoch=None):
        x, y = batch
        global Weight_quant_forward

        # if(prefix is 'test'):
        #     if(Weight_quant_forward == "ESA"):
        #         print("ESA : Now ternarizing for test)
        #         for name, module in self.backbone.named_modules():
        #             if isinstance(module,TernaryConv2d):
        #                 with torch.no_grad():
        #                     module.weight.data = ESA_Test_rounding(module.weight.data)
        #                 global_logger.log_parameter(name + " Ternarized",True)

        logits = self(x)
        preds = torch.argmax(logits, dim=1)  # predictions needed for metrics

        if(Weight_quant_forward == "ESA"):
            loss = self.loss(input=logits, target=y) 

            ESA_total = 0
            for name, module in self.backbone.named_modules():
                if isinstance(module,TernaryConv2d):
                    ESA_total = ESA_total + ESA_Loss_addition(module.weight)

            self.logger.experiment.log_metric(name = "Loss logger ESA", value=ESA_total, step=self.trainer.global_step, epoch=self.current_epoch)
            self.logger.experiment.log_metric(name = "Loss loggeer Normal", value=loss, step=self.trainer.global_step, epoch=self.current_epoch)
            loss = loss + ESA_total
        else:
            loss = self.loss(input=logits, target=y)


        #print(loss)


        self.log(f'{prefix}/loss', loss, on_step=on_step, on_epoch=on_epoch)
        self.log_dict(metric_collection(preds, y), on_step=on_step, on_epoch=on_epoch)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, 'training', self.train_metrics, on_step=True, on_epoch=True)

    def validation_step(self, batch, batch_idx):
        self._step(batch, batch_idx, 'validation', self.validation_metrics, on_epoch=True)

    def test_step(self, batch, batch_idx):
        self._step(batch, batch_idx, 'test', self.test_metrics, on_epoch=True)

    """
    def on_after_backward(self):
        for name, module in self.backbone.named_modules():
            if isinstance(module,TernaryConv2d):
                grad_set = module.weight.grad.cpu()
                self.logger.experiment.log_histogram_3d(grad_set, name=name, step=self.trainer.global_step, epoch=self.current_epoch)

    def on_fit_end(self):
        print("FIT has ended")

        global Weight_quant_forward
        if(Weight_quant_forward == "ESA"):
            print("ESA : Now ternarizing for test and val")
            for name, module in self.backbone.named_modules():
                if isinstance(module,TernaryConv2d):
                    module.weight = ESA_Test_rounding(module.weight)
                    global_logger.log_parameter(name + " Ternarized",True)
    """


    def on_epoch_end(self):

        Update_Epoch_Counter(self.current_epoch)



        for name, module in self.backbone.named_modules():
            if (isinstance(module, FATNN_Act_Quantizer)):
                alpha0 = module.alpha0
                alpha1 = module.alpha1
                self.logger.experiment.log_metric(name = name + " alpha0", value=alpha0, step=self.trainer.global_step, epoch=self.current_epoch)
                self.logger.experiment.log_metric(name = name + " alpha1", value=alpha1, step=self.trainer.global_step, epoch=self.current_epoch)
            if (isinstance(module, FATNN__Weight_Quantizer)):
                #self.logger.experiment.log_histogram_3d([module.alpha0,module.alpha1], name=name, step=self.trainer.global_step, epoch=self.current_epoch)
                alpha0 = module.alpha0
                alpha1 = module.alpha1
                self.logger.experiment.log_metric(name = name + " alpha0", value=alpha0, step=self.trainer.global_step, epoch=self.current_epoch)
                self.logger.experiment.log_metric(name = name + " alpha1", value=alpha1, step=self.trainer.global_step, epoch=self.current_epoch)
            if (isinstance(module, RSign_Ternary)):
                #self.logger.experiment.log_histogram_3d([module.alpha0,module.alpha1], name=name, step=self.trainer.global_step, epoch=self.current_epoch)
                bias0 = module.bias0
                bias1 = module.bias1
                self.logger.experiment.log_metric(name = name + " bias0", value=bias0, step=self.trainer.global_step, epoch=self.current_epoch)
                self.logger.experiment.log_metric(name = name + " bias1", value=bias1, step=self.trainer.global_step, epoch=self.current_epoch)
            #if isinstance(module,TernaryConv2d):
            #    grad_set = module.weight.cpu()
            #    self.logger.experiment.log_histogram_3d(grad_set, name=name, step=self.trainer.global_step, epoch=self.current_epoch)
            if isinstance(module, TTQ_Quant):
                w_p_cpy = module.W_p
                w_n_cpy = module.W_n
                self.logger.experiment.log_metric(name = name + "W_p", value=w_p_cpy, step=self.trainer.global_step, epoch=self.current_epoch)
                self.logger.experiment.log_metric(name = name + "W_n", value=w_n_cpy, step=self.trainer.global_step, epoch=self.current_epoch)




    # learning rate warm-up
    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, optimizer_closure, on_tpu, using_native_amp,using_lbfgs):
        warmup_epochs = self.hparams.warmup_epochs
        global lr
        initial_lr = float(lr)
        if epoch < warmup_epochs:
            lr_scale = min(1., (self.trainer.global_step + 1) / (warmup_epochs * self.trainer.num_training_batches))
            for pg in optimizer.param_groups:
                pg['lr'] = lr_scale * initial_lr

        # update params
        optimizer.step(closure=optimizer_closure)



    def configure_optimizers(self):
        global lr
        global Phase 
        global Two_Step


        optimizer = torch.optim.SGD(params=list(self.parameters()), lr=float(lr), momentum=0.9, dampening=0, weight_decay=1e-4, nesterov=False)

        if(Two_Step == "True"):
            self.trainer.max_epochs = int(self.trainer.max_epochs/2)
        else:
            self.trainer.max_epochs = self.trainer.max_epochs

        if(lr_scheduling == "Cosine"):
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=self.trainer.max_epochs, eta_min=0)
        elif(lr_scheduling == "Multi"):
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, gamma=0.1, milestones=[100, 150, 180])

        return [optimizer], [lr_scheduler]


# ------------
# DataModule
# ------------
# CHANGED: Inclusion of transforms (and autoaugment) and validation set equal to test set
class CIFAR10DataModule(pl_bolts.datamodules.cifar10_datamodule.CIFAR10DataModule):
    normalization = pl_bolts.transforms.dataset_normalizations.cifar10_normalization()

    def __init__(self, data_dir: str = './data/', val_split: int = 0, num_workers: int = 8, batch_size: int = 128,
                 autoaugment: bool = True):
        self.val_is_test = True if val_split == 0 else False

        default_transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), self.normalization])
        train_transforms = torchvision.transforms.Compose([
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomCrop(size=32, padding=4),
            torchvision.transforms.ToTensor(),
            self.normalization
        ])
        if autoaugment:
            train_transforms.transforms.insert(2, torchvision.transforms.AutoAugment(
                policy=torchvision.transforms.AutoAugmentPolicy.CIFAR10
            ))
        super().__init__(data_dir=data_dir, val_split=val_split, num_workers=num_workers, normalize=False,
                         batch_size=batch_size, #seed=random.randint(-2 ** 63, 2 ** 64 - 1),
                         shuffle=True, pin_memory=True, drop_last=False, train_transforms=train_transforms,
                         val_transforms=default_transforms, test_transforms=default_transforms)

    def setup(self, stage=None):
        if self.val_is_test:
            train_transforms = self.default_transforms() if self.train_transforms is None else self.train_transforms
            val_transforms = self.default_transforms() if self.val_transforms is None else self.val_transforms
            test_transforms = self.default_transforms() if self.test_transforms is None else self.test_transforms
            self.dataset_train = self.dataset_cls(self.data_dir, train=True, transform=train_transforms,
                                                  **self.EXTRA_ARGS)
            self.dataset_val = self.dataset_cls(self.data_dir, train=False, transform=val_transforms,
                                                **self.EXTRA_ARGS)
            self.dataset_test = self.dataset_cls(self.data_dir, train=False, transform=test_transforms,
                                                 **self.EXTRA_ARGS)
        else:
            super().setup(stage=stage)


#### CIFAR100 dataloader ?
class CIFAR100DataModule(pl.LightningDataModule):
    mean_cifar100 = (0.5074,0.4867,0.4411)
    std_cifar100 = (0.2011,0.1987,0.2025)
    normalization = tt.Normalize(mean=mean_cifar100, std= std_cifar100)

    def __init__(self, data_dir: str = './data/', num_workers: int = 8, batch_size: int = 128, autoaugment: bool = True):

        self.batch_size = batch_size
        self.data_dir = data_dir
        self.num_workers = num_workers
        self.autoaugment = autoaugment
        super().__init__()
            #data_dir=data_dir, val_split=val_split, num_workers=num_workers, normalize=False,
            #            batch_size=batch_size, #seed=random.randint(-2 ** 63, 2 ** 64 - 1),
            #             shuffle=True, pin_memory=True, drop_last=False, train_transforms=train_transforms,
            #             val_transforms=default_transforms, test_transforms=default_transforms)

    def setup(self, stage=None):
        default_transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), self.normalization])
        train_transforms = torchvision.transforms.Compose([
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomCrop(size=32, padding=4),
            torchvision.transforms.ToTensor(),
            self.normalization
            ])
        if self.autoaugment:
            train_transforms.transforms.insert(2, torchvision.transforms.AutoAugment(policy=torchvision.transforms.AutoAugmentPolicy.CIFAR10))

        self.cifar100_train = CIFAR100(download=True,root= self.data_dir,transform=train_transforms)
        self.cifar100_test = CIFAR100(root= self.data_dir,train=False,transform=default_transforms)
        
    def train_dataloader(self):
        return DataLoader(self.cifar100_train, batch_size=self.batch_size, num_workers= self.num_workers,pin_memory=True,shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.cifar100_test, batch_size=self.batch_size,num_workers= self.num_workers,pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.cifar100_test, batch_size=self.batch_size,num_workers= self.num_workers,pin_memory=True)




trainer_defaults = {'max_epochs': 200, 'precision': 32, 'gpus': 1, 'benchmark': True,  # 'deterministic': True,
                    'logger': {'class_path': 'pytorch_lightning.loggers.CometLogger',
                            'init_args': {'save_dir': 'logs/'}},
                    'callbacks': [pl.callbacks.ModelCheckpoint(monitor='validation/Accuracy', mode='max'),
                                pl.callbacks.LearningRateMonitor(logging_interval='step')]
                    }






def main() -> None:
    #Parse the input stuff 
    #Config = configparser.ConfigParser()
    #Config.read(str(sys.argv[1]))
    #load_config_file_parameters(Config)

    global Phase
    #with torch.autograd.set_detect_anomaly(True):
    #process.current_process()._config['tempdir'] =  '~/.tmp/'
    #print(process.current_process()._config['tempdir'])


    os.environ['COMET_API_KEY'] = 'nTfxGH61xCUKcetJYLmHYAxUx'
    os.environ['COMET_WORKSPACE'] = 'vishnuvivek'
    #os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


    class CLI(LightningCLI):
        # ------------
        # Experiment naming
        # ------------
        def add_arguments_to_parser(self, parser):
            parser.add_argument("--config_custom", default="config.cfg")

        def before_instantiate_classes(self):
            filename = self.config["config_custom"]
            Config = configparser.ConfigParser()
            Config.read(filename)
            load_config_file_parameters(Config)

        def before_fit(self):

            global seed 
            seed_everything(seed)
            torch.manual_seed(seed)
            global name

            if((Phase == 1)and(Two_Step == "True")): 
                self.trainer.logger.experiment.set_name(name+" Phase1")
            elif((Phase == 2)and(Two_Step == "True")): 
                self.trainer.logger.experiment.set_name(name+" Phase2")                
            else:
                self.trainer.logger.experiment.set_name(name)

            log_all_hyperparams(self.trainer.logger.experiment)
            global global_logger
            global_logger = self.trainer.logger.experiment



        # ------------
        # CometLogger specifics
        # ------------
        def after_fit(self):
            if isinstance(self.trainer.logger, pl.loggers.CometLogger):
                # Upload checkpoint files
                self.trainer.logger.experiment.log_asset_folder(
                    folder=os.path.join(self.trainer.logger.save_dir, self.trainer.logger.name,
                                        self.trainer.logger.version),
                    log_file_name=True,
                    recursive=True
                )

    # ------------
    # Fit
    # ------------
    # This first phase would be common between the two sessions 


    if(dataset_type == "cifar100"):
        print("Option with CIFAR100 selected")
        cli = CLI(BaseModel, CIFAR100DataModule, trainer_defaults=trainer_defaults, save_config_callback=None)
    else:
        print("Option with CIFAR10 selected")
        cli = CLI(BaseModel, CIFAR10DataModule, trainer_defaults=trainer_defaults, save_config_callback=None)

    
    # ------------
    # Test
    # ------------
    cli.trainer.test()

    if((Phase == 1)and(Two_Step == "True")): 
        cli.trainer.save_checkpoint(name+"_Phase1.ckpt")

        Phase = 2
        if(dataset_type == "cifar100"):
            cli = CLI(BaseModel, CIFAR100DataModule, trainer_defaults=trainer_defaults, save_config_callback=None)
        else:
            cli = CLI(BaseModel, CIFAR10DataModule, trainer_defaults=trainer_defaults, save_config_callback=None)
        cli.trainer.test()
        cli.trainer.save_checkpoint(name+".ckpt")
    
    else:
        cli.trainer.save_checkpoint(name+".ckpt")





if __name__ == "__main__":
    main()
