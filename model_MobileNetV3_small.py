
import torch
from torch import nn
from torchvision.models import mobilenet_v3_small
import torch.nn.functional as F

class CPM(nn.Module):
    def __init__(self, num_classes=200):
        super(CPM, self).__init__()
        self.num_classes = num_classes
        self.fc = nn.Linear(576, num_classes)

    def forward(self, x, flag, device):
        B, C, H, W = x.shape
        a = x
        x = x.view(B, C, H * W).permute(0, 2, 1).contiguous()
        h = x
        x = self.fc(x)
        logits = torch.softmax(x, dim=-1)
        max_val, max_ids = torch.max(logits, dim=-1)
        norm = F.normalize(max_val, p=2)

        # CPM
        p_r = torch.zeros([B, self.num_classes]).to(device)
        if flag == "train":
            for i in range(B):
                for j in range(H*W):
                    p_r[i][max_ids[i][j]] += norm[i][j]
            p_r = F.normalize(p_r, p=1)

        # CPML
        x = torch.einsum('bnc, bn->bnc', h, norm)
        x = x.permute(0, 2, 1).contiguous()
        x = x.view(B, C, H, W)
        x = x + a

        return x, p_r

class CPML(nn.Module):
    def __init__(self, num_classes=200):
        super(CPML, self).__init__()
        self.backbone = nn.Sequential(*list(mobilenet_v3_small(pretrained=True).children())[:-2])
        self.selector = CPM(num_classes=num_classes)

        self.layer14_pool = nn.AvgPool2d((14, 14), stride=1)

        self.layer15 = nn.Sequential(
            nn.Conv2d(576, 1024, 1, stride=1),
            nn.Hardswish(),
        )

        self.layer16 = nn.Sequential(
            nn.Conv2d(1024, num_classes, 1, stride=1),
        )

    def forward(self, x, flag="train"):
        x = self.backbone(x)
        x, p_r = self.selector(x, flag, x.device)
        x = self.layer14_pool(x)
        x = self.layer15(x)
        x = self.layer16(x)
        x = torch.squeeze(x)

        return x, p_r

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = CPML(num_classes=200)
    model = model
    inputs = torch.randn((2, 3, 448, 448))
    out, p_r = model(inputs)
    print(out.shape)




