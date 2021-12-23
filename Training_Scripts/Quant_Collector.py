import comet_ml
import os
import random
import pl_bolts
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics
import torchvision
from pytorch_lightning.utilities.cli import LightningCLI
from pytorch_lightning.utilities.seed import seed_everything
from torch import nn


epoch_counter = 1

####################################################################################################################################
######################################################  UTILS ######################################################################
####################################################################################################################################

def Update_Epoch_Counter(epoch):
    global epoch_counter
    epoch_counter = epoch
    return 0

def Load_matching_dicts(pretrained_dict, model):

    # Load current model 
    model_dict = model.state_dict()
    # Remove stuff from pretrained model that isnt there in the Current model
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

    # Put the pretrained weights in the current model 
    model_dict.update(pretrained_dict)
    # Model must now be loaded with the new state dict 
    model.load_state_dict(model_dict)
    print("Matching Dicts loaded")
    return model

####################################        Full precision fallback             ###################################################

class STE_FWD(nn.Module):
    def __init__(self):
        super(STE_FWD, self).__init__()

    def forward(input):
        return input

class STE_BWD(nn.Module):
    def __init__(self):
        super(STE_BWD, self).__init__()

    def forward(input,output):
        return output

######################        Element Wise Gradient Scale backwards pass (https://arxiv.org/abs/2104.00903)        ######################

class EWGS_grad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, output, input, EWGS_Factor):
        ctx.save_for_backward(output, input, EWGS_Factor)
        OUTPUT2 = output*1
        return OUTPUT2 

    @staticmethod
    def backward(ctx, grad_input):
        output, input, EWGS_Factor = ctx.saved_tensors

        scale = 1 + (EWGS_Factor)*torch.sign(grad_input)*(input-output)

        GRAD =  scale*grad_input

        return GRAD, None, None

def EWGS_Grad(output, input,  EWGS_Factor):
    EWGS_factor = torch.tensor(float(EWGS_Factor))
    output2 = EWGS_grad.apply(output, input, EWGS_factor)
    return output2

# def EWGS_grad(grad_input,FP_Input,Quant_Output,ewgs_scale, clip = True):

#     #print(grad_input)
#     #print("EWGS backwards eh")
#     scale = 1 + (ewgs_scale)*torch.sign(grad_input)*(FP_Input-Quant_Output)
#     Gradient = scale*grad_input

#     selector = FP_Input.gt(1.0) | FP_Input.lt(-1.0)
#     Gradient[selector] = 0

#     #print(Gradient)
#     return Gradient


####################################################################################################################################
#############################           Rounding methods with specific enhancements        #########################################
#############################                        Output = [0, 1]                       #########################################
#############################            Decision point = 0,  Input = (-inf , inf)         #########################################
####################################################################################################################################


class roundSTEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, delta):

        output = torch.where(input >  delta,  1.0, 0.0) 
        #ctx.save_for_backward(input, output)# delta)
        return output 

    @staticmethod
    def backward(ctx, grad_input):

        #input2, output2, delta = ctx.saved_tensors
        #print("delta :" + str(delta))
        #print("Input max/min values " + str(input2.max()) +" / "+str(input2.min()))
        #norm_input = (input2/(input2.abs()).max())
        #diff = input2 - output2
        #print("Diff valuye = " + str(diff.abs().max()))
        #scale = 1 + (10)*torch.sign(grad_input)*(diff)
        #print("Printing EWGS scale max value"+ str(scale.abs().max()))
        #GRAD =  scale*grad_input

        return grad_input, None

def round_ste(input, delta):
    return roundSTEFunction.apply(input, delta)

####################################################################################################################################

class Approx_Sign(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, delta):

        output = torch.where(input >  delta,  1.0, 0.0) 
        ctx.save_for_backward(input,output,delta)
        
        return output 


    @staticmethod
    def backward(ctx, grad_input):
        input,output,delta = ctx.saved_tensors

        x1_mask = ((input <= (delta))&(input > (delta-0.5))).type(torch.float32)
        x2_mask = ((input >  delta)&(input <  (delta+0.5))).type(torch.float32)

        x2 = (input - delta + 0.5)

        y1_dx = (4*x2)*x1_mask
        y2_dx = (4-4*x2)*x2_mask
        base = torch.tensor(0.25).cuda()

        grad_output = grad_input*((y1_dx+y2_dx)+base)

        return grad_output, None

def Approx_sign(input, delta):
    return Approx_Sign.apply(input, delta)
####################################################################################################################################


class gradual_exponent(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, delta):
       
        output = torch.where(input >  delta,  1.0, 0.0) 
        ctx.save_for_backward(input,output,delta)

        return output 

    @staticmethod
    def backward(ctx, grad_input):
        
        input,output,delta = ctx.saved_tensors

        i  = ((epoch_counter)//25)+1

        grad_output = ((i*(torch.exp(i*(input-delta))))/torch.pow((1+torch.exp(i*(input-delta))),2))*grad_input

        return grad_output, None


def Gradual_exponent(input, delta):
    return gradual_exponent.apply(input, delta)



####################################################################################################################################

class round_ste_selector(nn.Module):
    def __init__(self,round_type):
        super(round_ste_selector, self).__init__()

        if(round_type == "STE"):
            self.rounding = round_ste
            print("Rounding STE selected")
        elif(round_type == "Triangle"):
            self.rounding = Approx_sign
            print("Approx Sign selected")
        elif(round_type == "Gradual"):
            self.rounding = Gradual_exponent
            print("Gradual Hardening selected")
        else:
            self.rounding = round_ste

    def forward(self, input, delta):
        return self.rounding(input, delta)

####################################################################################################################################

class LearnableBias(nn.Module):
    def __init__(self, out_chn):
        super(LearnableBias, self).__init__()
        self.bias = nn.Parameter(torch.zeros(1,out_chn,1,1), requires_grad=True)

    def forward(self, x):
        out = x + self.bias.expand_as(x)
        return out

class RPreLU(nn.Module):
    def __init__(self,planes):
        super(RPreLU, self).__init__()
        print("RPReLU setup Complete")
        self.Rprelu = nn.Sequential( LearnableBias(planes), nn.PReLU(planes), LearnableBias(planes))


    def forward(self, input):
        return self.Rprelu(input)


####################################################################################################################################
####################################################################################################################################
#############################           Weight Quantization Methods           ######################################################
####################################################################################################################################
####################################################################################################################################


####################################################################################################################################
######################        Ternary Weight Networks forward pass (https://arxiv.org/abs/1605.04711)        ######################
####################################################################################################################################




def delta_calculation(input):
    output = (0.7*((torch.sum(torch.abs(input)))/torch.numel(input)))
    return output 

def alpha_calculation(input, delta):
    in_abs = torch.abs(input)
    Mask_1 = (in_abs > delta).float()
    number_subset = torch.sum(Mask_1)
    output = (torch.sum(in_abs*Mask_1)/number_subset)

    return output 



class TWN_Quant(nn.Module):
    def __init__(self,Weight_Round_Method, EWGS_Factor):
        super(TWN_Quant, self).__init__()
        self.round_method = round_ste_selector(Weight_Round_Method)
        self.EWGS_Factor = EWGS_Factor

    def forward(self, input):
        x = input 
        
        delta = delta_calculation(x)
        alpha = alpha_calculation(x, delta)


        #Mask_1 = x > (delta)
        #Mask_2 = x < (-1*delta)
        #Mask_1 = Mask_1.type(torch.float32)
        #Mask_2 = Mask_2.type(torch.float32)
        #output = (Mask_1*alpha) + (Mask_2*alpha*-1)


        Mask1 = self.round_method(x,delta)       #  0 to 1 range 
        Mask2 = self.round_method(x,-delta) - 1  # -1 to 0 range 

        output = (Mask1 + Mask2)*alpha

        output2 = EWGS_Grad( output, input, self.EWGS_Factor)

        return output2




####################################################################################################################################
######################        TRAINED TERNARY QUANTIZATION forward pass (https://arxiv.org/abs/1612.01064)        ##################
####################################################################################################################################



class TTQ_backward(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, w_p, w_n, t, output):

        ctx.save_for_backward(input, w_p, w_n, t)

        return output


    @staticmethod
    def backward(ctx, grad_input):

        input, w_p, w_n, t = ctx.saved_tensors

        delta1 = t*(input.max())
        delta2 = t*(input.min())

        Mask1 = (input > delta1).float()
        Mask2 = (input < delta2).float()
        Mask3 = torch.ones(input.size()).cuda() - Mask1 - Mask2


        # scaled kernel grad and grads for scaling factors (w_p, w_n)
        return w_p.abs()*Mask1*grad_input + w_n.abs()*Mask2*grad_input + 1.0*Mask3*grad_input,  (Mask1*grad_input).sum(),  (Mask2*grad_input).sum(), None, None


class TTQ_Quant(nn.Module):
    def __init__(self,Weight_Round_Method, EWGS_Factor):
        super(TTQ_Quant, self).__init__()

        self.W_p = nn.Parameter(torch.tensor(1.0), requires_grad=True)
        self.W_n = nn.Parameter(torch.tensor(1.0), requires_grad=True)
        self.t   = nn.Parameter(torch.tensor(0.05), requires_grad=False)
        self.round_method = round_ste_selector(Weight_Round_Method)
        self.EWGS_Factor = EWGS_Factor

    def forward(self, input):

        x = input
        delta1 = self.t*(x.max())
        delta2 = self.t*(x.min())
        Mask1 = self.round_method(x, delta1)      #  0 to 1 range 
        Mask2 = self.round_method(x, delta2) - 1  # -1 to 0 range 

        output = (Mask1*self.W_p.abs()) + (self.W_n.abs()*Mask2)

        output2 = TTQ_backward.apply(input, self.W_p, self.W_n, self.t, output)

        output3 = EWGS_Grad(output2, input, self.EWGS_Factor)


        return output3

####################################################################################################################################
#######        Ternary Neural Networks with Fine-Grained Quantization forward pass (https://arxiv.org/abs/1705.01462)        #######
####################################################################################################################################


def delta_FGQ(input):
    output = (0.7*((torch.sum(torch.abs(input)))/torch.numel(input)))
    #output = ( (torch.sum(torch.abs(input)))/torch.numel(input) ) 
    return output 

def alpha_FGQ(input, output):

    sizing = input.size()
    num = sizing[0]*sizing[1]
    a = torch.sum(input.abs(),[0,1])
    alpha = a/num

    return output[:,:]*alpha

class FGQ_Quant(nn.Module):

    def __init__(self,Weight_Round_Method, EWGS_Factor):
        super(FGQ_Quant, self).__init__()
        self.round_method = round_ste_selector(Weight_Round_Method)
        self.EWGS_Factor = EWGS_Factor

    def forward(self, input):
        x = input

        # Calculate the deltas on FGQ specified 
        delta = delta_FGQ(x)

        Mask1 = self.round_method(x, delta)      #  0 to 1 range 
        Mask2 = self.round_method(x, -1*delta) - 1  # -1 to 0 range 

        output = (Mask1 + Mask2)

        output2 = alpha_FGQ(x, output)

        output3 = EWGS_Grad( output2, input, self.EWGS_Factor)

        return output3




####################################################################################################################################
###########       Embarrassingly Simple Approach Ternary Weight forward pass (https://arxiv.org/abs/2011.00580)        #############
####################################################################################################################################

def ESA_Loss_addition(weight):
        # alpha = 0.1 , lambada = 5e-6
        alpha= 0.1
        reg_input=0.000005
        # Normal cross entropy loss 
        # ESA LOss
        sq_weight = torch.pow( torch.tanh(weight), 2)

        # (alpha-W^2)* W^2
        ESA_loss = reg_input*(((alpha - sq_weight)*sq_weight).sum())
        
        return ESA_loss

class ESA_Quant(nn.Module):
    def __init__(self,Weight_Round_Method, EWGS_Factor):
        super(ESA_Quant, self).__init__()
        self.round_method = round_ste_selector(Weight_Round_Method)
        self.EWGS_Factor = EWGS_Factor

    def forward(self, input):

        W_theta = torch.tanh(input)

        W_new = W_theta*3

        Mask1 = self.round_method(W_new, torch.tensor(0.5))      #  0 to 1 range 
        Mask2 = self.round_method(W_new, torch.tensor(-0.5)) - 1  # -1 to 0 range
        output = Mask1 + Mask2
        
        output2 = EWGS_Grad(output, W_new, self.EWGS_Factor)

        return output2



####################################################################################################################################
#############################################      FATNN Weight quatizer    ########################################################
####################################################################################################################################

class FATNN__Weight_Quantizer(nn.Module):
    def __init__(self,Weight_Round_Method, EWGS_Factor):
        super(FATNN__Weight_Quantizer, self).__init__()
        self.alpha0 = nn.Parameter(torch.tensor(1.0), requires_grad=True)
        self.alpha1 = nn.Parameter(torch.tensor(1.0), requires_grad=True)
        self.quant = round_ste_selector(Weight_Round_Method )
        self.clamp = torch.clamp
        self.EWGS_Factor = EWGS_Factor

    def forward(self, input):
        alpha0 = self.alpha0.abs()
        alpha1 = self.alpha1.abs()
        y1 = input #* alpha0
        #y1 = self.clamp(y1, min=-1, max=0)
        y1 = self.quant(y1,-0.5*alpha0) - 1
        y2 = input #* alpha1
        #y2 = self.clamp(y2, min=0, max=1)
        y2 = self.quant(y2, 0.5*alpha1)
        output = y1 + y2

        output2 = EWGS_Grad(output, input, self.EWGS_Factor)

        return output2






####################################################################################################################################
####################################################################################################################################
#############################           Activation Quantization Methods           ##################################################
####################################################################################################################################
####################################################################################################################################




####################################################################################################################################
############################################      FATNN Activation quatizer    #####################################################
####################################################################################################################################



class FATNN_Act_Quantizer(nn.Module):
    def __init__(self,Activation_Round_Method, EWGS_Factor):
        super(FATNN_Act_Quantizer, self).__init__()
        self.alpha0 = nn.Parameter(torch.tensor(1.0), requires_grad=True)
        self.alpha1 = nn.Parameter(torch.tensor(1.0), requires_grad=True)
        self.quant = round_ste_selector(Activation_Round_Method)
        self.clamp = torch.clamp
        self.EWGS_Factor = EWGS_Factor

    def forward(self, input):

        alpha0 = self.alpha0.abs()
        y1 = input * alpha0
        y1 = self.clamp(y1, min=0, max=1)
        y1 = self.quant(y1, torch.tensor(0.5))

        alpha1 = self.alpha1.abs()
        y2 = (input - 1.0/alpha0) * alpha1
        y2 = self.clamp(y2, min=0, max=1)
        y2 = self.quant(y2, torch.tensor(0.5))

        output = y1 + y2
        
        output2 = EWGS_Grad(output, input.detach(), self.EWGS_Factor)

        return output2




####################################################################################################################################
#############################################      RSign Activation quatizer    ####################################################
####################################################################################################################################

class RSign_Ternary(nn.Module):
    def __init__(self,Activation_Round_Method, EWGS_Factor):
        super(RSign_Ternary, self).__init__()
        self.bias0 = nn.Parameter(torch.tensor(0.5), requires_grad=True)
        self.bias1 = nn.Parameter(torch.tensor(1.0), requires_grad=True)
        self.quant = round_ste_selector(Activation_Round_Method)
        self.clamp = torch.clamp
        self.EWGS_Factor = EWGS_Factor
        
    def forward(self, input):

        bias0 = self.bias0.abs()
        bias1 = self.bias1.abs()
        x1 = input - bias0
        y1 = self.clamp(x1, min=-2*bias0.item(),max=10**-5)
        out1 = self.quant(y1, -0.5*bias0)
        x2 = input - bias1
        y2 = self.clamp(x2, min=10**-5,max=2*bias1.item())
        out2 = self.quant(y2,1.5*bias1)

        #x2 = self.bias1(input)
        #print("IS this even logged")
        #plt.figure()
        #sns.distplot(input.cpu().flatten())
        # sns.set_style('darkgrid')
        # sns.displot(torch.flatten(input.cpu())).set(title = "Quant input Distribution")

        output = out1 + out2
        

        #sns.distplot(output.cpu().flatten())

        #output2 = RSIGN_backward.apply(output, bias0, bias1, input)

        output2 = EWGS_Grad(output, input, self.EWGS_Factor)

        # sns.set_style('darkgrid')
        # sns.displot(torch.flatten(output2.cpu())).set(title = "Convolution output Distribution")

        return output2


