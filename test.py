import torch
from composition_models import ComposeAE

model = ComposeAE("", 10, 20, True, "")
x = torch.Tensor(2, 3, 10, 10).fill_(1.).cuda()
y = torch.Tensor(2, 3, 20, 20).fill_(2.).cuda()
model = model.cuda()

z = model.compose_img_text(x, y)
print(z)