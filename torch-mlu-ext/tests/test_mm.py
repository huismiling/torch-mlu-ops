
import torch
import torch_mlu

from mlu_ext.functions import mm


mm_dtype = torch.int8
a = torch.rand(128, 512).to(mm_dtype)
b = torch.rand(512, 256).to(mm_dtype)

if mm_dtype in [torch.float, torch.float16]:
    a = a.clone().detach().requires_grad_(True)
    b = b.clone().detach().requires_grad_(True)

    a_mlu = a.mlu().clone().detach().requires_grad_(True)
    b_mlu = b.mlu().clone().detach().requires_grad_(True)
else:
    a_mlu = a.mlu().clone().to(torch.int16)
    b_mlu = b.mlu().clone()

c = torch.mm(a, b)
d = mm(a_mlu, b_mlu)
diff = d.detach().cpu() - c

print(f"a: {a.dtype}, b: {b.dtype}")
print(f"a_mlu: {a_mlu.dtype}, b_mlu: {b_mlu.dtype}")
print(f"d: {d.dtype}, c: {c.dtype}")
print(diff.max(), diff.mean(), diff.sum())

assert torch.allclose(a_mlu.to(torch.int16), a.mlu().to(torch.int16))
assert torch.allclose(b_mlu.to(torch.int16), b.mlu().to(torch.int16))
assert torch.allclose(d.detach().half(), c.detach().mlu().half(), rtol=0.003)

if mm_dtype in [torch.float, torch.float16]:
    d.mean().backward()
    c.mean().backward()

    print(a.grad.mean(), a.grad.view(-1)[:10])
    print(a_mlu.grad.mean(), a_mlu.grad.view(-1)[:10])

    print(b_mlu.grad.mean(), b_mlu.grad.view(-1)[:10])
    print(b.grad.mean(), b.grad.view(-1)[:10])

    diff = d.detach().cpu() - c.detach()

    print(diff.max(), diff.mean(), diff.sum())

    assert torch.allclose(d.detach(), c.detach().mlu(), rtol=0.003)
    assert torch.allclose(a_mlu.grad, a.grad.mlu(), rtol=0.003)
    assert torch.allclose(b_mlu.grad, b.grad.mlu(), rtol=0.003)

