

import torch
from torch import nn
from torchvision.models import resnet50
import torch.nn.functional as F


class CPM(nn.Module):
    def __init__(self, num_classes=200):
        super(CPM, self).__init__()
        self.num_classes = num_classes
        self.fc = nn.Linear(2048, num_classes)

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
            # p_r = torch.softmax(p_r, dim=-1)
            p_r = F.normalize(p_r, p=1)

        # main
        x = torch.einsum('bnc, bn->bnc', h, norm)
        x = x.permute(0, 2, 1).contiguous()
        x = x.view(B, C, H, W)
        x = x + a

        return x, p_r

class CPML(nn.Module):
    def __init__(self, num_classes=200):
        super(CPML, self).__init__()
        self.backbone = nn.Sequential(*list(resnet50(pretrained=True).children())[:-2])
        self.selector = CPM(num_classes=num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x, flag="train"):
        x = self.backbone(x)
        x, p_r = self.selector(x, flag, x.device)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x, p_r

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = CPML(num_classes=200)
    # model_weight_path = "./result/CUB200/best_model_RA.pth"
    # model.load_state_dict(torch.load(model_weight_path))
    model = model.to(device)
    # print(model)
    inputs = torch.randn((2, 3, 448, 448)).to(device)
    out, _ = model(inputs, flag="train")
    # print(model)
    print(out.shape)