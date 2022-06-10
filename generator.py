

    def forward(self, x):
        block1 = self.prelu(self.conv1(x))
        block2 = torch.add(
            self.bn(self.conv2(self.prelu(self.bn(self.conv2(block1))))), block1)
        block3 = torch.add(
            self.bn(self.conv2(self.prelu(self.bn(self.conv2(block2))))), block2)
        block4 = torch.add(
            self.bn(self.conv2(self.prelu(self.bn(self.conv2(block3))))), block3)
        block5 = torch.add(
            self.bn(self.conv2(self.prelu(self.bn(self.conv2(block4))))), block4)
        block6 = torch.add(
            self.bn(self.conv2(self.prelu(self.bn(self.conv2(block5))))), block5)
        block7 = torch.add(self.bn(self.conv2(block6)), block1)
        block8 = self.prelu(self.ps(self.conv3_1(block7)))
        block9 = self.prelu(self.ps(self.conv3_2(block8)))
        block10 = self.conv4(block9)
        return block10

        gen = Generator().to(cuda)


# Uncomment below mentioned three lines if you have more than one gpu and want to use all of them
# ngpu=2
# if (cuda.type == 'cuda') and (ngpu > 1):
#     gen = n.DataParallel(gen, list(range(ngpu)))
summary(gen, (3, 64, 64))
