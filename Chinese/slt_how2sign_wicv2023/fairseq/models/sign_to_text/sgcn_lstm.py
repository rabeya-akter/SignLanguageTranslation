import torch.nn as nn
import torch
import torch.nn.functional as F
from fairseq.models.sign_to_text.graph import Graph




class sgcn(nn.Module):
    def __init__(self, AD, AD2,bias_mat_1, bias_mat_2, batch_size=10):
        super(sgcn, self).__init__()
        self.AD = AD
        self.AD2 = AD2
        self.batch_size = batch_size
        self.num_joints = 33
        self.bias_mat_1 = bias_mat_1
        self.bias_mat_2 = bias_mat_2
        self.bias_mat_1.to('cuda:0')
        self.bias_mat_2.to('cuda:0')



        #self.conv_lstm1=ConvLSTM(64,33,(1,1),1,True, True, False)
        #self.conv_lstm2=ConvLSTM(64,33,(1,1),1,True, True, False)
        self.dropout = nn.Dropout2d(p=0.25)
        #self.bias_mat_1=torch.zeros((33,33)).to('cuda:0')
        #self.bias_mat_2=torch.zeros((33,33)).to('cuda:0')

    def forward(self, Input):
        k1 = F.relu(nn.Conv2d(Input.shape[1], 64, kernel_size=(9, 1), padding=(4, 0)).to('cuda:0')(Input))

        k = torch.cat((Input, k1), dim=1)

        x1 = F.relu(nn.Conv2d(Input.shape[1] + k1.shape[1], 64, kernel_size=(1, 1)).to('cuda:0')(k))
        
        gcn_x1 = torch.einsum('vw,ncwt->ncvt', (self.AD, x1))


        y1 = F.relu(nn.Conv2d(k.shape[1], 64, kernel_size=(1, 1)).to('cuda:0')(k))
        gcn_y1 = torch.einsum('vw,ncwt->ncvt', (self.AD2, y1))

        gcn_1 = torch.cat((gcn_x1, gcn_y1), dim=1)

        z1 = F.relu(nn.Conv2d(gcn_1.shape[1], 16, kernel_size=(9, 1), padding=(4, 0)).to('cuda:0')(gcn_1))
        z1 = self.dropout(z1)

        z2 = F.relu(nn.Conv2d(z1.shape[1], 16, kernel_size=(15, 1), padding=(7, 0)).to('cuda:0')(z1))
        z2 = self.dropout(z2)

        z3 = F.relu(nn.Conv2d(z2.shape[1], 16, kernel_size=(19, 1), padding=(9, 0)).to('cuda:0')(z2))
        z3 = self.dropout(z3)

        z = torch.cat((z1, z2, z3), dim=1)

        return z

class Sgcn_Lstm(nn.Module):
    def __init__(self, train_x=None, train_y=None, AD=None, AD2=None, lr=0.0001, epoach=200, batch_size=10):
        super(Sgcn_Lstm, self).__init__()
        #self.train_x = train_x
        #self.train_y = train_y
        #self.AD = AD
        #self.AD2 = AD2
        #self.lr = lr
        #self.epoach =epoach
        #self.batch_size = batch_size
        #self.num_joints = 543
        graph=Graph(33)



        self.sgcn_1=sgcn(graph.AD, graph.AD2, graph.bias_mat_1, graph.bias_mat_2)
        self.sgcn_2=sgcn(graph.AD, graph.AD2, graph.bias_mat_1, graph.bias_mat_2)
        self.sgcn_3=sgcn(graph.AD, graph.AD2, graph.bias_mat_1, graph.bias_mat_2)

    def lstm(self,x):
        x = x.view(x.shape[0], -1, x.shape[1] * x.shape[2])
        rec,_ = nn.LSTM(input_size=x.shape[2], hidden_size=256, num_layers=1, batch_first=True, dropout=0.25, bidirectional=False).to('cuda:0')(x)
        """
        rec,_ = nn.LSTM(input_size=x.shape[2], hidden_size=64, num_layers=1, batch_first=True, dropout=0.25, bidirectional=True).to('cuda:0')(x)
        rec1,_ = nn.LSTM(input_size=rec.shape[-1], hidden_size=32, num_layers=1, batch_first=True, dropout=0.25, bidirectional=True).to('cuda:0')(rec)
        rec2,_ = nn.LSTM(input_size=rec1.shape[-1], hidden_size=32, num_layers=1, batch_first=True, dropout=0.25, bidirectional=True).to('cuda:0')(rec1)
        rec3,_ = nn.LSTM(input_size=rec2.shape[-1], hidden_size=256, num_layers=1, batch_first=True, dropout=0.25, bidirectional=False).to('cuda:0')(rec2)
        """
        out=rec
        return out


    def forward(self, x):
        x = x.permute(0,3,2,1)
        x = self.sgcn_1(x)
        y = self.sgcn_2(x)
        y = y + x
        z = self.sgcn_3(y)
        z = z + y
        """
        batch_size, _, _, frame_size = z.size()
        z = z.view(batch_size * frame_size, -1)
        z=nn.Linear(48 * 33, 256).to('cuda:0')(z)
        z=z.view(batch_size,frame_size,-1)
        #print('z: ',z.shape)
        out=z
        """

        out = self.lstm(z)
        return out