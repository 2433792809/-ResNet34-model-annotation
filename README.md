ResNet34模型注释
——————————————————————————————————————————
-------------------------------------------------------------------------------------------------------------------------------------------------------------

🔸搭建模型的模板
---------------------------------------------------------------------------------------------------------------------------------------------------------
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    class xxxNet(nn.Module):
    def "#init"#(self):
    pass
    def forward(x):
    return x
    
 出现的语法问题：
 
 forward 方法的定义中缺少了 self 参数，因为在类方法中必须包含 self 参数，以便访问对象的属性和方法
 
 会导致的后果：
 
 Python 解释器将无法识别您想要访问的成员，并会导致程序出错
 
---------------------------------------------------------------------------------------------------------------------------------------------------------

🔸 ResNet34 模型代码注释
---------------------------------------------------------------------------------------------------------------------------------------------------------
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    #把残差连接补充到 Block 的 forward 函数中
    class Block(nn.Module):
            def __init__(self, dim, out_dim, stride) -> None:
            super().__init__()
            self.conv1 = nn.Conv2d(dim, out_dim, kernel_size=3, stride=stride, padding=1)
            self.bn1 = nn.BatchNorm2d(out_dim)
            self.relu1 = nn.ReLU()
            self.conv2 = nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1)
            self.bn2 = nn.BatchNorm2d(out_dim)
            self.relu2 = nn.ReLU()

        def forward(self, x):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu1(x)
            x = self.conv2(x)
            x = self.bn2(x)
            x = self.relu2(x)
            return x


    class ResNet32(nn.Module):
        def __init__(self, in_channel=64, num_classes=2):
            super().__init__()
            self.num_classes = num_classes
            self.in_channel = in_channel

            self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2, padding=3)
            self.maxpooling = nn.MaxPool2d(kernel_size=2)
            self.last_channel = in_channel

            self.layer1 = self._make_layer(in_channel=64, num_blocks=3, stride=1)
            self.layer2 = self._make_layer(in_channel=128, num_blocks=4, stride=2)
            self.layer3 = self._make_layer(in_channel=256, num_blocks=6, stride=2)
            self.layer4 = self._make_layer(in_channel=512, num_blocks=3, stride=2)

            self.avgpooling = nn.AvgPool2d(kernel_size=2)
            self.classifier = nn.Linear(4608, self.num_classes)

        def _make_layer(self, in_channel, num_blocks, stride):
            layer_list = [Block(self.last_channel, in_channel, stride)]
            self.last_channel = in_channel
            for i in range(1, num_blocks):
                b = Block(in_channel, in_channel, stride=1)
                layer_list.append(b)
            return nn.Sequential(*layer_list)

        def forward(self, x):
            x = self.conv1(x)  # [bs, 64, 56, 56] 特征提取过程
            x = self.maxpooling(x)  # [bs, 64, 28, 28]池化，降低分辨率和计算量
            x = self.layer1(x)  # [bs, 256, 28, 28]
            x = self.layer2(x)  # [bs, 512, 14, 14]
            x = self.layer3(x)  # [bs, 1024, 7, 7]
            x = self.layer4(x)  # [bs, 2048, 7, 7]
            x = self.avgpooling(x)  # [bs, 2048, 1, 1]，全局平均池化层对张量进行降维处理
            x = x.view(x.shape[0], -1)  # [bs, 2048]，将张量展开成一维，方便后续全连接层的计算
            x = self.classifier(x)  # [bs, num_classes]，全连接层对特征进行分类
            output = F.softmax(x)  # [bs, num_classes]，对输出张量进行归一化处理

            return output


    if __name__=='__main__':
        t = torch.randn([8, 3, 224, 224])
        model = ResNet32()
        out = model(t)
        print(out.shape)
