import torch
import torch.nn as nn

class block(nn.Module):
    
    def __init__(self,in_channels,out_channels,id_downsample=None,stride=1):
        super(block,self).__init__()
        self.expansion=4
        self.conv1=nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=1,padding=0)
        self.bn1=nn.BatchNorm2d(out_channels)
        self.conv2=nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=stride,padding=1)
        self.bn2=nn.BatchNorm2d(out_channels)
        self.conv3=nn.Conv2d(out_channels,out_channels*self.expansion,kernel_size=1,stride=1,padding=0)
        self.bn3=nn.BatchNorm2d(out_channels*self.expansion)
        self.relu=nn.ReLU()
        self.id_downsample=id_downsample

    def forward(self,X):
        identity=X

        X=self.conv1(X)
        X=self.bn1(X)
        X=self.relu(X)
        X=self.conv2(X)
        X=self.bn2(X)
        X=self.relu(X)
        X=self.conv3(X)
        X=self.bn3(X)

        if self.id_downsample is not None:
            identity=self.id_downsample(identity)

        X+=identity
        X=self.relu(X)
        return X


class ResNet(nn.Module): #[3,4,6,3]

    def __init__(self,block,layers,img_channels,n_classes):
        super(ResNet,self).__init__()
        self.in_channels=64
        self.conv1=nn.Conv2d(img_channels,64,kernel_size=7,stride=2,padding=3)
        self.bn1=nn.BatchNorm2d(64)
        self.relu=nn.ReLU()
        self.maxpool=nn.MaxPool2d(kernel_size=3,stride=2,padding=1)

        # ResNet layers:
        self.layer1=self._make_layers(block,layers[0],out_channels=64,stride=1)
        self.layer2=self._make_layers(block,layers[1],out_channels=128,stride=2)
        self.layer3=self._make_layers(block,layers[2],out_channels=256,stride=2)
        self.layer4=self._make_layers(block,layers[3],out_channels=512,stride=2)

        self.avgpool=nn.AdaptiveAvgPool2d((1,1))
        self.fc=nn.Linear(512*4,n_classes)
        
    def forward(self,X):
        X=self.conv1(X)
        X=self.bn1(X)
        X=self.relu(X)
        X=self.maxpool(X)
        X=self.layer1(X)
        X=self.layer2(X)
        X=self.layer3(X)
        X=self.layer4(X)

        X=self.avgpool(X)
        X=X.reshape(X.shape[0],-1)
        X=self.fc(X)

        return X
        

    def _make_layers(self,block,n_residual_blocks,out_channels,stride):
        id_downsample=None
        layers=[]

        if stride!= 1 or self.in_channels!= out_channels*4:
            id_downsample=nn.Sequential(nn.Conv2d(self.in_channels,out_channels*4,kernel_size=1,stride=stride),nn.BatchNorm2d(out_channels*4))

        layers.append(block(self.in_channels,out_channels,id_downsample,stride))
        self.in_channels=out_channels*4

        for i in range(n_residual_blocks-1):
            layers.append(block(self.in_channels,out_channels))

        return nn.Sequential(*layers)




        
        
        
        
        

        
        
        
        
    