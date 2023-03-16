from torch import nn

class ResMod(nn.Module):
    def __init__(self,c,k_s) -> None:
        super(ResMod,self).__init__()
        self.conv1 = nn.Conv2d(c,c,k_s,padding=int((k_s-1)/2))
        self.r1 = nn.ReLU()
        self.conv2 = nn.Conv2d(c,c,k_s,padding=int((k_s-1)/2))
        self.r2 = nn.ReLU()

    def forward(self,x):
        x1 = self.conv1(x)
        x1 = self.r1(x1)
        x1 = self.conv2(x1)
        return self.r2(x1 + x)

class Modle(nn.Module):
    def __init__(self) -> None:
        super(Modle,self).__init__()
        self.conv_1 = nn.Sequential(
            nn.Conv2d(3,16,7,2,3),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv_2 = nn.Sequential(
            nn.Conv2d(16,32,5,2,2),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv_3 = nn.Sequential(
            nn.Conv2d(32,64,3,1,1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.resmod = nn.Sequential(
            *[ResMod(64,3) for i in range(40)]
        )
        self.liner = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4096,512),
            nn.Linear(512,512),
            nn.Linear(512,2),
        )

    def forward(self,input):
        x = self.conv_1(input)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = self.resmod(x)
        x = self.liner(x)
        return x