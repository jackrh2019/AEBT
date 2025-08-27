import torch
import matplotlib.pyplot as plt

def norm2img(x, mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)):
    device = x.device
    mean = torch.tensor(mean, device=device)[None,:,None,None]
    std  = torch.tensor(std,  device=device)[None,:,None,None]
    return (x * std + mean).clamp(0,1)

def visualize_batch_3views(model, loader, device='cuda', n=30, mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)):
    """
    可视化 x1, x2, x3, x1_rec, x2_rec, x3_rec 及 z1, z2, z3, zb1, zb2, zb3 向量折线图，每行单独输出。
    """
    model.eval()
    # 拿到第一个 batch
    (views, _), _ = next(iter(loader)), None
    x1 = views[0][:n].to(device)
    x2 = views[1][:n].to(device)
    x3 = views[2][:n].to(device)

    with torch.no_grad():
        z1, zb1, t1, _, x1_rec = model(x1)
        z2, zb2, t2, _, x2_rec = model(x2)
        z3, zb3, t3, _, x3_rec = model(x3)

    x1_img     = norm2img(x1,     mean, std)
    x2_img     = norm2img(x2,     mean, std)
    x3_img     = norm2img(x3,     mean, std)
    x1_rec_img = x1_rec.clamp(0,1)
    x2_rec_img = x2_rec.clamp(0,1)
    x3_rec_img = x3_rec.clamp(0,1)

#    titles = ['x1','x1_rec','x2','x2_rec','x3','x3_rec','z1','z2','z3','zb1','zb2','zb3']
    for i in range(n):
        fig, axes = plt.subplots(1, 12, figsize=(24, 2.4))
        # 图像部分
        axes[0].imshow(x1_img[i].cpu().permute(1,2,0));    axes[0].axis('off')
        axes[1].imshow(x1_rec_img[i].cpu().permute(1,2,0));axes[1].axis('off')
        axes[2].imshow(x2_img[i].cpu().permute(1,2,0));    axes[2].axis('off')
        axes[3].imshow(x2_rec_img[i].cpu().permute(1,2,0));axes[3].axis('off')
        axes[4].imshow(x3_img[i].cpu().permute(1,2,0));    axes[4].axis('off')
        axes[5].imshow(x3_rec_img[i].cpu().permute(1,2,0));axes[5].axis('off')
        # 向量折线图
        axes[6].plot(z1[i].cpu().numpy());  axes[6].set_xlim(0, z1.size(1))
        axes[7].plot(z2[i].cpu().numpy());  axes[7].set_xlim(0, z2.size(1))
        axes[8].plot(z3[i].cpu().numpy());  axes[8].set_xlim(0, z3.size(1))
        axes[9].plot(zb1[i].cpu().numpy()); axes[9].set_xlim(0, zb1.size(1))
        axes[10].plot(zb2[i].cpu().numpy());axes[10].set_xlim(0, zb2.size(1))
        axes[11].plot(zb3[i].cpu().numpy());axes[11].set_xlim(0, zb3.size(1))
        # 只在第一行加标题
#        for j, t in enumerate(titles):
#            axes[j].set_title(t, fontsize=12)
        plt.tight_layout()
        plt.show()