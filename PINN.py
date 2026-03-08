import torch
import torch.nn as nn
import matplotlib.pyplot as plt


"""
1.定义PINN神经网络 输入层+4个隐藏层+输出层，均为全连接层
2.生成数据点 
3.定义损失函数  三个约束
4.定义训练函数  常规，加入调节器plateau
5.主程序可视化结果
"""

class PINN(nn.Module):
    def __init__(self,hidden_layers=4,hidden_size=50):
        super(PINN,self).__init__()
        self.input_layer = nn.Linear(2,hidden_size)   #输入层
        self.hidden_layers = nn.ModuleList([
            nn.Linear(hidden_size,hidden_size) for _ in range(hidden_layers)
        ])    #4个隐藏层
        self.output_layer = nn.Linear(hidden_size,1)   #输出层
        self.activation = nn.Tanh()   #光滑激活函数，生成连续解

    #神经网络
    def forward(self,x,t):
        input = torch.cat([x,t],1)
        out = self.activation(self.input_layer(input))
        for layer in self.hidden_layers:
            out = self.activation(layer(out))
        return self.output_layer(out)

#生成训练数据
def generate_data():
    # 初始条件（IC）： t=0 , x∈[0,2π]
    x_ic = torch.linspace(0,2 * torch.pi,100).reshape(-1,1)  #[100,1]
    t_ic = torch.zeros_like(x_ic) #[100,1]
    # 边界条件（BC）： x=0和x=2π, t∈[0,1]
    t_bc = torch.linspace(0, 1, 50).reshape(-1, 1)
    x_0 = torch.zeros_like(t_bc)
    x_2pi = 2 * torch.pi * torch.ones_like(t_bc)
    # PDE残差点: x∈[0,2π], t∈[0,1]（随机采样）
    x_pde = torch.rand(500, 1) * 2 * torch.pi
    t_pde = torch.rand(500, 1)
    return (x_ic, t_ic), (x_0, t_bc, x_2pi, t_bc), (x_pde, t_pde)

# 计算损失（包含IC/BC/PDE约束）
def compute_loss(model, x_ic, t_ic, x_0, t_0, x_2pi, t_2pi, x_pde, t_pde):
    # 初始条件损失 L_IC
    u_ic_pred = model(x_ic, t_ic)
    u_ic_true = torch.sin(x_ic - torch.pi / 2) + 1
    L_IC = torch.mean((u_ic_pred - u_ic_true) ** 2)

    # 边界条件损失 L_BC
    u_bc0_pred = model(x_0, t_0)
    u_bc2pi_pred = model(x_2pi, t_2pi)
    L_BC = torch.mean(u_bc0_pred ** 2) + torch.mean(u_bc2pi_pred ** 2)

    # PDE残差损失 L_PDE
    x_pde.requires_grad_(True)
    t_pde.requires_grad_(True)
    u_pde = model(x_pde, t_pde)
    u_t = torch.autograd.grad(u_pde.sum(), t_pde, create_graph=True)[0]  # ∂u/∂t  加和对t求偏导生成元组再取第一个元素为关于t的张量列表
    u_x = torch.autograd.grad(u_pde.sum(), x_pde, create_graph=True)[0]  # ∂u/∂x
    u_xx = torch.autograd.grad(u_x.sum(), x_pde, create_graph=True)[0]  # ∂²u/∂x²
    R = u_t - u_xx  # 热方程残差
    L_PDE = torch.mean(R ** 2)

    return L_IC + L_BC + L_PDE, L_IC, L_BC, L_PDE


#训练  Adam为优化器  加入Plateau作为调节器
def train_model(model, epochs=30000, lr=1e-4):
    (x_ic, t_ic), (x_0, t_0, x_2pi, t_2pi), (x_pde, t_pde) = generate_data()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2000
    )

    loss_history = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        total_loss, L_IC, L_BC, L_PDE = compute_loss(
            model, x_ic, t_ic, x_0, t_0, x_2pi, t_2pi, x_pde, t_pde
        )
        total_loss.backward()
        optimizer.step()
        scheduler.step(total_loss.item())  # 使用 .item() 获取标量值

        loss_history.append(total_loss.item())
        if (epoch + 1) % 1000 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch + 1}: Loss={total_loss:.6f}, "
                  f"IC={L_IC:.6f}, BC={L_BC:.6f}, PDE={L_PDE:.6f}, "
                  f"LR={current_lr:.2e}")
    return loss_history

# 5. 主程序：训练+可视化
if __name__ == "__main__":
    # 初始化并训练模型
    model = PINN(hidden_layers=4, hidden_size=50)
    loss_history = train_model(model, epochs=30000)

    # 可视化损失曲线
    plt.figure(figsize=(8, 4))
    plt.plot(loss_history)
    plt.xlabel("Epoch")
    plt.ylabel("Total Loss")
    plt.title("PINN Training Loss")
    plt.show()

    # 可视化不同时刻的解
    x_test = torch.linspace(0, 2 * torch.pi, 200).reshape(-1, 1)
    t_list = [0.0, 0.25, 0.5, 0.75, 1.0]
    plt.figure(figsize=(10, 6))
    for t in t_list:
        t_tensor = t * torch.ones_like(x_test)
        u_pred = model(x_test, t_tensor).detach().numpy()
        plt.plot(x_test.numpy(), u_pred, label=f"t={t}")
    # 叠加初始条件
    u_ic_true = torch.sin(x_test - torch.pi / 2) + 1
    plt.plot(x_test.numpy(), u_ic_true.numpy(), "k--", label="t=0 (True)")
    plt.xlabel("x")
    plt.ylabel("u(x,t)")
    plt.title("PINN Solution of Heat Equation")
    plt.legend()
    plt.show()
