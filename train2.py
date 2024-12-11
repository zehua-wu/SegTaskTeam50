
















optimizer = optim.AdamW(model.parameters(),
                        lr=1e-3, 
                        weight_decay=1e-4)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
