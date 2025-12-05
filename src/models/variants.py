import torch
import torch.nn as nn
import torch.nn.functional as F

class batchnorm_1d(nn.Module):
    '''
    1d Convolutional layers

    Arguments:
        num_in_filters {int} -- number of input filters
        num_out_filters {int} -- number of output filters
        kernel_size {tuple} -- size of the convolving kernel
        stride {tuple} -- stride of the convolution (default: {(1, 1)})
        activation {str} -- activation function (default: {'relu'})

    '''
    def __init__(self, num_in_filters, num_out_filters, kernel_size, stride = 1, activation = 'relu'):
        super().__init__()
        self.activation = activation
        self.conv1 = nn.Conv1d(in_channels=num_in_filters, out_channels=num_out_filters, kernel_size=kernel_size, stride=stride, padding = 'same')
        self.batchnorm = nn.BatchNorm1d(num_out_filters)
    
    def forward(self,x):
        x = self.conv1(x)
        x = self.batchnorm(x)
        
        if self.activation == 'relu':
            return F.relu(x)
        else:
            return x


class Multiresblock(nn.Module):
    '''
    MultiRes Block
    
    Arguments:
        num_in_channels {int} -- Number of channels coming into mutlires block
        num_filters {int} -- Number of filters in a corrsponding UNet stage
        alpha {float} -- alpha hyperparameter (default: 1.67)
    
    '''

    def __init__(self, num_in_channels, num_filters, alpha=1.67):
    
        super().__init__()
        self.alpha = alpha
        self.W = num_filters * alpha
        
        filt_cnt_3x3 = int(self.W*0.167)
        filt_cnt_5x5 = int(self.W*0.333)
        filt_cnt_7x7 = int(self.W*0.5)
        num_out_filters = filt_cnt_3x3 + filt_cnt_5x5 + filt_cnt_7x7
        
        self.shortcut = batchnorm_1d(num_in_channels ,num_out_filters , kernel_size = 1, activation='None')

        self.conv_3x3 = batchnorm_1d(num_in_channels, filt_cnt_3x3, kernel_size = 3, activation='relu')

        self.conv_5x5 = batchnorm_1d(filt_cnt_3x3, filt_cnt_5x5, kernel_size = 3, activation='relu')
        
        self.conv_7x7 = batchnorm_1d(filt_cnt_5x5, filt_cnt_7x7, kernel_size = 3, activation='relu')

        self.batch_norm1 = nn.BatchNorm1d(num_out_filters)
        self.batch_norm2 = nn.BatchNorm1d(num_out_filters)

    def forward(self,x):

        shrtct = self.shortcut(x)
        
        a = self.conv_3x3(x)
        b = self.conv_5x5(a)
        c = self.conv_7x7(b)

        x = torch.cat([a,b,c],axis=1)
        x = self.batch_norm1(x)

        x = x + shrtct
        x = self.batch_norm2(x)
        x = nn.functional.relu(x)
    
        return x


class Respath(nn.Module):
    '''
    ResPath
    
    Arguments:
        num_in_filters {int} -- Number of filters going in the respath
        num_out_filters {int} -- Number of filters going out the respath
        respath_length {int} -- length of ResPath
        
    '''

    def __init__(self, num_in_filters, num_out_filters, respath_length):
    
        super().__init__()

        self.respath_length = respath_length
        self.shortcuts = nn.ModuleList([])
        self.convs = nn.ModuleList([])
        self.bns = nn.ModuleList([])

        for i in range(self.respath_length):
            if(i==0):
                self.shortcuts.append(batchnorm_1d(num_in_filters, num_out_filters, kernel_size = 1, activation='None'))
                self.convs.append(batchnorm_1d(num_in_filters, num_out_filters, kernel_size = 3,activation='relu'))

                
            else:
                self.shortcuts.append(batchnorm_1d(num_out_filters, num_out_filters, kernel_size = 1, activation='None'))
                self.convs.append(batchnorm_1d(num_out_filters, num_out_filters, kernel_size = 3, activation='relu'))

            self.bns.append(nn.BatchNorm1d(num_out_filters))
        
    
    def forward(self,x):

        for i in range(self.respath_length):

            shortcut = self.shortcuts[i](x)

            x = self.convs[i](x)
            x = self.bns[i](x)
            x = nn.functional.relu(x)

            x = x + shortcut
            x = self.bns[i](x)
            x = nn.functional.relu(x)

        return x

class AttentionBlock(nn.Module):
    def __init__(self,F_g,F_l,F_int):
        super(AttentionBlock,self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv1d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm1d(F_int)
            )
        
        self.W_x = nn.Sequential(
            nn.Conv1d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm1d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv1d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm1d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self,g,x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)

        return x*psi

class MultiResUnet(nn.Module):
    '''
    MultiResUNet
    
    Arguments:
        input_channels {int} -- number of channels in image
        num_classes {int} -- number of segmentation classes
        alpha {float} -- alpha hyperparameter (default: 1.67)
    
    Returns:
        [Tensorflow model] -- MultiResUNet model
    '''
    def __init__(self, input_channels=1, num_classes=1, alpha=1.67):
        super().__init__()
        
        self.alpha = alpha
        
        # Encoder Path
        self.multiresblock1 = Multiresblock(input_channels,32)
        self.in_filters1 = int(32*self.alpha*0.167)+int(32*self.alpha*0.333)+int(32*self.alpha* 0.5)
        self.pool1 =  nn.MaxPool1d(2)
        self.respath1 = Respath(self.in_filters1,32,respath_length=4)

        self.multiresblock2 = Multiresblock(self.in_filters1,32*2)
        self.in_filters2 = int(32*2*self.alpha*0.167)+int(32*2*self.alpha*0.333)+int(32*2*self.alpha* 0.5)
        self.pool2 =  nn.MaxPool1d(2)
        self.respath2 = Respath(self.in_filters2,32*2,respath_length=3)
    
    
        self.multiresblock3 =  Multiresblock(self.in_filters2,32*4)
        self.in_filters3 = int(32*4*self.alpha*0.167)+int(32*4*self.alpha*0.333)+int(32*4*self.alpha* 0.5)
        self.pool3 =  nn.MaxPool1d(2)
        self.respath3 = Respath(self.in_filters3,32*4,respath_length=2)
    
    
        self.multiresblock4 = Multiresblock(self.in_filters3,32*8)
        self.in_filters4 = int(32*8*self.alpha*0.167)+int(32*8*self.alpha*0.333)+int(32*8*self.alpha* 0.5)
        self.pool4 =  nn.MaxPool1d(2)
        self.respath4 = Respath(self.in_filters4,32*8,respath_length=1)
    
    
        self.multiresblock5 = Multiresblock(self.in_filters4,32*16)
        self.in_filters5 = int(32*16*self.alpha*0.167)+int(32*16*self.alpha*0.333)+int(32*16*self.alpha* 0.5)
     
        # Decoder path
        self.upsample6 = nn.ConvTranspose1d(self.in_filters5,32*8,kernel_size=2,stride=2)  
        self.concat_filters1 = 32*8 *2
        self.multiresblock6 = Multiresblock(self.concat_filters1,32*8)
        self.in_filters6 = int(32*8*self.alpha*0.167)+int(32*8*self.alpha*0.333)+int(32*8*self.alpha* 0.5)
        self.attention1 = AttentionBlock(F_g=self.in_filters6,F_l=32*8,F_int=self.in_filters6//2)

        self.upsample7 = nn.ConvTranspose1d(32*8,32*4,kernel_size=2,stride=2)  
        self.concat_filters2 = 32*4 *2
        self.multiresblock7 = Multiresblock(self.concat_filters2,32*4)
        self.in_filters7 = int(32*4*self.alpha*0.167)+int(32*4*self.alpha*0.333)+int(32*4*self.alpha* 0.5)
        self.attention2 = AttentionBlock(F_g=self.in_filters7,F_l=32*4,F_int=self.in_filters7//2)
    
        self.upsample8 = nn.ConvTranspose1d(32*4,32*2,kernel_size=2,stride=2)
        self.concat_filters3 = 32*2 *2
        self.multiresblock8 = Multiresblock(self.concat_filters3,32*2)
        self.in_filters8 = int(32*2*self.alpha*0.167)+int(32*2*self.alpha*0.333)+int(32*2*self.alpha* 0.5)
        self.attention3 = AttentionBlock(F_g=self.in_filters8,F_l=32*2,F_int=self.in_filters8//2)
    
        self.upsample9 = nn.ConvTranspose1d(32*2,32,kernel_size=2,stride=2)
        self.concat_filters4 = 32 *2
        self.multiresblock9 = Multiresblock(self.concat_filters4,32)
        self.in_filters9 = int(32*self.alpha*0.167)+int(32*self.alpha*0.333)+int(32*self.alpha* 0.5)
        self.attention4 = AttentionBlock(F_g=self.in_filters9,F_l=32,F_int=self.in_filters9//2)

        self.conv_final = batchnorm_1d(32, num_classes, kernel_size = 1, activation='None')

    def forward(self,x : torch.Tensor)->torch.Tensor:

        x_multires1 = self.multiresblock1(x)
        x_pool1 = self.pool1(x_multires1)
        x_multires1 = self.respath1(x_multires1)
        
        x_multires2 = self.multiresblock2(x_pool1)
        x_pool2 = self.pool2(x_multires2)
        x_multires2 = self.respath2(x_multires2)

        x_multires3 = self.multiresblock3(x_pool2)
        x_pool3 = self.pool3(x_multires3)
        x_multires3 = self.respath3(x_multires3)

        x_multires4 = self.multiresblock4(x_pool3)
        x_pool4 = self.pool4(x_multires4)
        x_multires4 = self.respath4(x_multires4)

        x_multires5 = self.multiresblock5(x_pool4)

        up6 = torch.cat([self.upsample6(x_multires5),x_multires4],axis=1)
        x_multires6 = self.multiresblock6(up6)
        x_multires6 = self.attention1(x_multires6,x_multires4)
        

        up7 = torch.cat([self.upsample7(x_multires6),x_multires3],axis=1)
        x_multires7 = self.multiresblock7(up7)
        x_multires7 = self.attention2(x_multires7,x_multires3)

        up8 = torch.cat([self.upsample8(x_multires7),x_multires2],axis=1)
        x_multires8 = self.multiresblock8(up8)
        x_multires8 = self.attention3(x_multires8,x_multires2)

        up9 = torch.cat([self.upsample9(x_multires8),x_multires1],axis=1)
        x_multires9 = self.multiresblock9(up9)
        x_multires9 = self.attention4(x_multires9,x_multires1)

        out =  self.conv_final(x_multires9)
        
        return out


class GeneratorUNet(nn.Module):
    def __init__(self, input_channels=1, output_channels=1):
        super(GeneratorUNet, self).__init__()

        def down_block(in_feat, out_feat, normalize=True):
            layers = [nn.Conv1d(in_feat, out_feat, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2))
            return layers

        def up_block(in_feat, out_feat, dropout=0.0):
            layers = [
                nn.ConvTranspose1d(in_feat, out_feat, 4, stride=2, padding=1),
                nn.BatchNorm1d(out_feat, 0.8),
                nn.ReLU(inplace=True),
            ]
            if dropout:
                layers.append(nn.Dropout(dropout))
            return layers

        # Encoder
        self.down1 = nn.Sequential(*down_block(input_channels, 64, normalize=False))
        self.down2 = nn.Sequential(*down_block(64, 128))
        self.down3 = nn.Sequential(*down_block(128, 256))
        self.down4 = nn.Sequential(*down_block(256, 512, normalize=False)) # Bottleneck

        # Decoder
        self.up1 = nn.Sequential(*up_block(512, 256))
        self.up2 = nn.Sequential(*up_block(512, 128)) # Skip connection 256+256=512
        self.up3 = nn.Sequential(*up_block(256, 64))  # Skip connection 128+128=256
        
        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv1d(128, output_channels, 3, stride=1, padding=1), # Skip connection 64+64=128
            nn.Tanh()
        )

    def forward(self, x):
        # x: (B, 1, 512)
        d1 = self.down1(x) # (B, 64, 256)
        d2 = self.down2(d1) # (B, 128, 128)
        d3 = self.down3(d2) # (B, 256, 64)
        d4 = self.down4(d3) # (B, 512, 32)
        
        u1 = self.up1(d4) # (B, 256, 64)
        u1 = torch.cat((u1, d3), dim=1) # (B, 512, 64)
        
        u2 = self.up2(u1) # (B, 128, 128)
        u2 = torch.cat((u2, d2), dim=1) # (B, 256, 128)
        
        u3 = self.up3(u2) # (B, 64, 256)
        u3 = torch.cat((u3, d1), dim=1) # (B, 128, 256)
        
        out = self.final(u3) # (B, 1, 512)
        return out


