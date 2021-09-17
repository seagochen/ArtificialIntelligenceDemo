import torch

model = torch.load("./LSTM_Surname_Classfication_CPU_98per.ptm")

# converting to Torch Script via Annotation
serialized_model = torch.jit.script(model)

# save the torch script for C++
serialized_model.save("LSTM_Surname_Classfication_CPU_98per.pt")