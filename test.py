import torch

with torch.no_grad():
    # 模型初始化
    linear1 = torch.nn.Linear(1024,1024, bias=False).cuda() # + 4194304(4096KB)
    print(torch.cuda.memory_allocated()/(1024))   #4194304   4096KB
    linear2 = torch.nn.Linear(1024, 1, bias=False).cuda() # + 4096(4KB)
    print(torch.cuda.memory_allocated()/(1024))   #4198400   4100KB

    # 输入定义
    inputs = torch.tensor([[1.0]*1024]*1024).cuda() # shape = (1024,1024) # + 4194304(4096KB)
    print(torch.cuda.memory_allocated()/(1024))   #8392704   8196KB

    # 前向传播
    # loss = sum(linear2(linear1(inputs)))  #shape = (1) # memory + 4194304(4096KB) + 512(0.5KB)
    # print(torch.cuda.memory_allocated()/(1024))   #21107200  20612KB
    y = linear1(inputs)
    print("cache: %r" % (torch.cuda.memory_reserved()/1024))
    print(torch.cuda.memory_allocated()/(1024))
    print(f"max: {torch.cuda.max_memory_allocated()/1024}")
    u = linear2(y)
    print(torch.cuda.memory_allocated()/(1024))
    loss = sum(u)
    print(torch.cuda.memory_allocated()/(1024))

# # 后向传播
# loss.backward() # memory - 4194304 + 4194304 + 4096
# print(torch.cuda.memory_allocated()/(1024))

# # 再来一次~
# loss = sum(linear2(linear1(inputs))) # shape = (1) # memory + 4194304  (512没了，因为loss的ref还在)
# print(torch.cuda.memory_allocated()/(1024))
# loss.backward() # memory - 4194304
# print(torch.cuda.memory_allocated()/(1024))