import torch
import torch.nn as nn
from torchvision import transforms
from torchsummary import summary

class Stage_down(nn.Module):
    def __init__(self, chan_in, chan_out, kernel_size = 3, acti_func=nn.ReLU(inplace=True)):
        super().__init__()
        #The two convolutions
        self.conv1 = nn.Conv2d(chan_in, chan_out, kernel_size, padding="same")
        self.conv2 = nn.Conv2d(chan_out, chan_out, kernel_size, padding="same")

        #The Maxpooling to go down
        self.maxpool2d = nn.MaxPool2d(2)

        #activation function
        self.activate = acti_func

        #We keep the value of the last cropped part:
        self.crop = None

    def forward(self, x):
        x = self.conv1(x)
        x = self.activate(x)
        x = self.conv2(x)
        x = self.activate(x)
    
        #Add the cropped value
        self.crop = x

        #Do the maxpooling
        x = self.maxpool2d(x)
        return x

class Stage_up(torch.nn.Module):
    def __init__(self, chan_in, chan_out, kernel_size = 3, acti_func=nn.ReLU(inplace=True)):
        super().__init__()
        
        #The two convolutions
        self.conv1 = nn.Conv2d(chan_in *2 , chan_in, kernel_size, padding="same")
        self.conv2 = nn.Conv2d(chan_in, chan_in, kernel_size, padding="same")

        #The up-convolution:
        self.up_conv = nn.ConvTranspose2d(chan_in, chan_out, 2, 2)

        #The activation function
        self.activate = acti_func

    def forward(self, x):
        
        x = self.conv1(x)
        x = self.activate(x)
        x = self.conv2(x)
        x = self.activate(x)
        x = self.up_conv(x)
        return x

class Unet(nn.Module):
    def __init__(self, channel_in, factor = 8, acti_func=nn.ReLU(inplace=True), channel_out=1, end_softmax=False):
        super().__init__()
        channel_fact = factor
        
        #The activation function
        self.activate = acti_func
        #Goign down
        self.stg1_down = Stage_down(channel_in, channel_fact, acti_func=acti_func)
        self.stg2_down = Stage_down(channel_fact, channel_fact*2, acti_func=acti_func)
        self.stg3_down = Stage_down(channel_fact*2, channel_fact*4, acti_func=acti_func)
        self.stg4_down = Stage_down(channel_fact*4, channel_fact*8, acti_func=acti_func)
        self.stg5_down = Stage_down(channel_fact*8, channel_fact*16, acti_func=acti_func)

        #Bottom:
        self.bottom1 = nn.Conv2d(channel_fact*16, channel_fact*32, kernel_size= 3, padding="same")
        self.bottom2 = nn.Conv2d(channel_fact*32, channel_fact*32, kernel_size= 3, padding="same")
        self.bottom_upconv = nn.ConvTranspose2d(channel_fact*32, channel_fact*16, 2, 2)

        #Going up
        self.stg1_up = Stage_up(channel_fact*16, channel_fact*8, acti_func=acti_func)
        self.stg2_up = Stage_up(channel_fact*8, channel_fact*4, acti_func=acti_func)
        self.stg3_up = Stage_up(channel_fact*4, channel_fact*2, acti_func=acti_func)
        self.stg4_up = Stage_up(channel_fact*2, channel_fact, acti_func=acti_func)

        #Top:
        self.top1 = nn.Conv2d(channel_fact*2, channel_fact, kernel_size=3, padding="same")
        self.top2 = nn.Conv2d(channel_fact, channel_fact, kernel_size=3, padding="same")

        self.final = nn.Conv2d(channel_fact, channel_out, kernel_size=1)
        
        self.softmax = nn.Identity()
        if end_softmax:
            self.softmax = nn.Softmax2d()
      

    def forward(self, x):
        
        original_shape = (x.shape[2], x.shape[3])
        #Going Down
        x = self.stg1_down(x)
        x = self.stg2_down(x)
        x = self.stg3_down(x)
        x = self.stg4_down(x)
        x = self.stg5_down(x)

        #Bottom
        x = self.bottom1(x)
        x = self.activate(x)
        x = self.bottom2(x)
        x = self.activate(x)
        x = self.bottom_upconv(x)

        #Going Up
        x = transforms.Resize((self.stg5_down.crop.shape[2], self.stg5_down.crop.shape[3]))(x)
        x = torch.cat((x, self.stg5_down.crop), dim=1)
        x = self.stg1_up(x)
        x = transforms.Resize((self.stg4_down.crop.shape[2], self.stg4_down.crop.shape[3]))(x)
        x = torch.cat((x, self.stg4_down.crop), dim=1)
        x = self.stg2_up(x)
        x = transforms.Resize((self.stg3_down.crop.shape[2], self.stg3_down.crop.shape[3]))(x)
        x = torch.cat((x, self.stg3_down.crop), dim=1)
        x = self.stg3_up(x)
        x = transforms.Resize((self.stg2_down.crop.shape[2], self.stg2_down.crop.shape[3]))(x)
        x = torch.cat((x, self.stg2_down.crop), dim=1)
        x = self.stg4_up(x)
        x = transforms.Resize((self.stg1_down.crop.shape[2], self.stg1_down.crop.shape[3]))(x)
        x = torch.cat((self.stg1_down.crop, x), dim=1)
        
        #Top
        x = self.top1(x)
        x = self.activate(x)
        x = self.top2(x)
        x = self.activate(x)
        x = transforms.Resize(original_shape)(x)
        x = self.final(x)
        x = self.softmax(x)
        return x



if __name__ == "__main__":
    
    model = Unet(1, 8).cuda()
    summary(model, (1, 400, 400))