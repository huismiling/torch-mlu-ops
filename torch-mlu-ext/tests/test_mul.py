
import torch
import torch_mlu

from mlu_ext.functions import mul_element



a = torch.rand(20, 30, 40).float()
b = torch.rand(20, 30, 40).float()

a = a.clone().detach().requires_grad_(True)
b = b.clone().detach().requires_grad_(True)

a_mlu = a.mlu().clone().detach().requires_grad_(True)
b_mlu = b.mlu().clone().detach().requires_grad_(True)

c = a*b
d = mul_element(a_mlu, b_mlu)

d.mean().backward()
c.mean().backward()

print(a.grad.mean(), a.grad.view(-1)[:10])
print(a_mlu.grad.mean(), a_mlu.grad.view(-1)[:10])

print(b_mlu.grad.mean(), b_mlu.grad.view(-1)[:10])
print(b.grad.mean(), b.grad.view(-1)[:10])

diff = d.detach().cpu() - c

print(diff.max(), diff.mean(), diff.sum())

