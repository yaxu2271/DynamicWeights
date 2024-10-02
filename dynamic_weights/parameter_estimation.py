import numpy as np
from pdb import set_trace as st
import matplotlib.pyplot as plt
import matplotlib
from model import FCN_poro_as_one
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from custom_loss import poro_loss
import scipy.io
import pandas as pd
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


# CUDA support
print('Initializing Network...')

if torch.cuda.is_available():
    device = torch.device('cuda')
    print('run with GPU')
else:
    device = torch.device('cpu')
    print('run with CPU')

def PrepareData(data_path):

    data = scipy.io.loadmat(data_path)


    x_star = data['xx']
    y_star = data['yy']
    ip_star = data['ip']
    ip_x_star = data['ip_x']
    ip_xx_star = data['ip_xx']
    ip_y_star = data['ip_y']
    ip_yy_star = data['ip_yy']
    iux_star = data['iux']
    iux_x_star = data['iux_x']
    iux_xx_star = data['iux_xx']
    iux_xy_star = data['iux_xy']
    iux_yy_star = data['iux_yy']
    iuy_star = data['iuy']
    iuy_xx_star = data['iuy_xx']
    iuy_y_star = data['iuy_y']
    iuy_yx_star = data['iuy_yx']
    iuy_yy_star = data['iuy_yy']
    rp_star = data['rp']
    rp_x_star = data['rp_x']
    rp_xx_star = data['rp_xx']
    rp_y_star = data['rp_y']
    rp_yy_star = data['rp_yy']
    rux_star = data['rux']
    rux_x_star = data['rux_x']
    rux_xx_star = data['rux_xx']
    rux_xy_star = data['rux_xy']
    rux_yy_star = data['rux_yy']
    ruy_star = data['ruy']
    ruy_xx_star = data['ruy_xx']
    ruy_y_star = data['ruy_y']
    ruy_yx_star = data['ruy_yx']
    ruy_yy_star = data['ruy_yy']

    rF1x_star = data['rF1x']
    iF1x_star = data['iF1x']
    rF1y_star = data['rF1y']
    iF1y_star = data['iF1y']
    rF2_star = data['rF2']
    iF2_star = data['iF2']




    x_starr = torch.from_numpy(np.array(x_star.flatten())).type(torch.FloatTensor)
    y_starr = torch.from_numpy(np.array(y_star.flatten())).type(torch.FloatTensor)
    ip_starr = torch.from_numpy(np.array(ip_star.flatten())).type(torch.FloatTensor)
    ip_x_starr = torch.from_numpy(np.array(ip_x_star.flatten())).type(torch.FloatTensor)
    ip_xx_starr = torch.from_numpy(np.array(ip_xx_star.flatten())).type(torch.FloatTensor)
    ip_y_starr = torch.from_numpy(np.array(ip_y_star.flatten())).type(torch.FloatTensor)
    ip_yy_starr = torch.from_numpy(np.array(ip_yy_star.flatten())).type(torch.FloatTensor)
    iux_starr = torch.from_numpy(np.array(iux_star.flatten())).type(torch.FloatTensor)
    iux_x_starr = torch.from_numpy(np.array(iux_x_star.flatten())).type(torch.FloatTensor)
    iux_xx_starr = torch.from_numpy(np.array(iux_xx_star.flatten())).type(torch.FloatTensor)
    iux_xy_starr = torch.from_numpy(np.array(iux_xy_star.flatten())).type(torch.FloatTensor)
    iux_yy_starr = torch.from_numpy(np.array(iux_yy_star.flatten())).type(torch.FloatTensor)
    iuy_starr = torch.from_numpy(np.array(iuy_star.flatten())).type(torch.FloatTensor)
    iuy_xx_starr = torch.from_numpy(np.array(iuy_xx_star.flatten())).type(torch.FloatTensor)
    iuy_y_starr = torch.from_numpy(np.array(iuy_y_star.flatten())).type(torch.FloatTensor)
    iuy_yx_starr = torch.from_numpy(np.array(iuy_yx_star.flatten())).type(torch.FloatTensor)
    iuy_yy_starr = torch.from_numpy(np.array(iuy_yy_star.flatten())).type(torch.FloatTensor)
    rp_starr = torch.from_numpy(np.array(rp_star.flatten())).type(torch.FloatTensor)
    rp_x_starr = torch.from_numpy(np.array(rp_x_star.flatten())).type(torch.FloatTensor)
    rp_xx_starr = torch.from_numpy(np.array(rp_xx_star.flatten())).type(torch.FloatTensor)
    rp_y_starr = torch.from_numpy(np.array(rp_y_star.flatten())).type(torch.FloatTensor)
    rp_yy_starr = torch.from_numpy(np.array(rp_yy_star.flatten())).type(torch.FloatTensor)
    rux_starr = torch.from_numpy(np.array(rux_star.flatten())).type(torch.FloatTensor)
    rux_x_starr = torch.from_numpy(np.array(rux_x_star.flatten())).type(torch.FloatTensor)
    rux_xx_starr = torch.from_numpy(np.array(rux_xx_star.flatten())).type(torch.FloatTensor)
    rux_xy_starr = torch.from_numpy(np.array(rux_xy_star.flatten())).type(torch.FloatTensor)
    rux_yy_starr = torch.from_numpy(np.array(rux_yy_star.flatten())).type(torch.FloatTensor)
    ruy_starr = torch.from_numpy(np.array(ruy_star.flatten())).type(torch.FloatTensor)
    ruy_xx_starr = torch.from_numpy(np.array(ruy_xx_star.flatten())).type(torch.FloatTensor)
    ruy_y_starr = torch.from_numpy(np.array(ruy_y_star.flatten())).type(torch.FloatTensor)
    ruy_yx_starr = torch.from_numpy(np.array(ruy_yx_star.flatten())).type(torch.FloatTensor)
    ruy_yy_starr = torch.from_numpy(np.array(ruy_yy_star.flatten())).type(torch.FloatTensor)

    rF1x_starr = torch.from_numpy(np.array(rF1x_star.flatten())).type(torch.FloatTensor)
    iF1x_starr = torch.from_numpy(np.array(iF1x_star.flatten())).type(torch.FloatTensor)
    rF1y_starr = torch.from_numpy(np.array(rF1y_star.flatten())).type(torch.FloatTensor)
    iF1y_starr = torch.from_numpy(np.array(iF1y_star.flatten())).type(torch.FloatTensor)
    rF2_starr = torch.from_numpy(np.array(rF2_star.flatten())).type(torch.FloatTensor)
    iF2_starr = torch.from_numpy(np.array(iF2_star.flatten())).type(torch.FloatTensor)




    return x_starr, y_starr, ip_starr, ip_x_starr, ip_xx_starr, ip_y_starr, ip_yy_starr, iux_starr, iux_x_starr, iux_xx_starr, iux_xy_starr, iux_yy_starr, iuy_starr, iuy_xx_starr, iuy_y_starr, iuy_yx_starr, iuy_yy_starr, rp_starr, rp_x_starr, rp_xx_starr, rp_y_starr, rp_yy_starr, rux_starr, rux_x_starr, rux_xx_starr, rux_xy_starr, rux_yy_starr, ruy_starr, ruy_xx_starr, ruy_y_starr, ruy_yx_starr, ruy_yy_starr, rF1x_starr, iF1x_starr, rF1y_starr, iF1y_starr, rF2_starr, iF2_starr


def main():
    MODE = 'train'


    if not os.path.exists('./logs/'):
        os.mkdir('./logs/')


    print('Preparing dataset:')
    x_train1, y_train1, ip_train1, ip_x_train1, ip_xx_train1, ip_y_train1, ip_yy_train1, iux_train1, iux_x_train1, iux_xx_train1, iux_xy_train1, iux_yy_train1, iuy_train1, iuy_xx_train1, iuy_y_train1, iuy_yx_train1, iuy_yy_train1, rp_train1, rp_x_train1, rp_xx_train1, rp_y_train1, rp_yy_train1, rux_train1, rux_x_train1, rux_xx_train1, rux_xy_train1, rux_yy_train1, ruy_train1, ruy_xx_train1, ruy_y_train1, ruy_yx_train1, ruy_yy_train1, rF1x_train1, iF1x_train1, rF1y_train1, iF1y_train1, rF2_train1, iF2_train1 = PrepareData(data_path='data/poro_dataF_source1_normalized_based_on_rp.mat')


    EPOCH = 20000
    LEARNING_RATE = 5e-3





    input_size_NN =40*40*2

    using_model = FCN_poro_as_one(input_size=input_size_NN).to(device)



    if MODE == 'train':


        print("Training Mode")
        optimizer = optim.Adam(list(using_model.parameters()), lr=LEARNING_RATE, betas=(0.9,0.9999),eps=1e-6)

        mu_parameter_list = []
        lambda_parameter_list = []
        M_list = []
        phi_list = []
        kappa_list = []
        alpha_list = []

        e1_loss_list = []
        e2_loss_list = []
        e3_loss_list = []
        e4_loss_list = []
        e5_loss_list = []
        e6_loss_list = []


        w1_list = []
        w2_list = []
        w3_list = []
        w4_list = []
        w5_list = []
        w6_list = []

        regularizer_loss_list = []

        loss_list = []





        plt.ion()
        for epoch in tqdm(range(1,EPOCH+1)):



            input_ = torch.cat([x_train1, y_train1]).to(device)





            output1, output2, output3, output4, output5, output6 = using_model(input_)




            loss, e1_loss, e2_loss, e3_loss, e4_loss, e5_loss, e6_loss, w1, w2, w3, w4, w5, w6 = poro_loss(ip_train1.to(device),
                                        ip_x_train1.to(device),
                                        ip_xx_train1.to(device),
                                        ip_y_train1.to(device),
                                        ip_yy_train1.to(device),
                                        iux_train1.to(device),
                                        iux_x_train1.to(device),
                                        iux_xx_train1.to(device),
                                        iux_xy_train1.to(device),
                                        iux_yy_train1.to(device),
                                        iuy_train1.to(device),
                                        iuy_xx_train1.to(device),
                                        iuy_y_train1.to(device),
                                        iuy_yx_train1.to(device),
                                        iuy_yy_train1.to(device),
                                        rp_train1.to(device),
                                        rp_x_train1.to(device),
                                        rp_xx_train1.to(device),
                                        rp_y_train1.to(device),
                                        rp_yy_train1.to(device),
                                        rux_train1.to(device),
                                        rux_x_train1.to(device),
                                        rux_xx_train1.to(device),
                                        rux_xy_train1.to(device),
                                        rux_yy_train1.to(device),
                                        ruy_train1.to(device),
                                        ruy_xx_train1.to(device),
                                        ruy_y_train1.to(device),
                                        ruy_yx_train1.to(device),
                                        ruy_yy_train1.to(device),
                                        rF1x_train1.to(device),
                                        iF1x_train1.to(device),
                                        rF1y_train1.to(device),
                                        iF1y_train1.to(device),
                                        rF2_train1.to(device),
                                        iF2_train1.to(device),
                                        output1,
                                        output2,
                                        output3,
                                        output4,
                                        output5,
                                        output6,)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            print(f'-------------Epoch {epoch}---------------')
            print(f'Loss:{loss.detach().cpu().numpy()}, e1 loss:{e1_loss.detach().cpu().numpy()}, e2 loss:{e2_loss.detach().cpu().numpy()}, e3 loss:{e3_loss.detach().cpu().numpy()}, e4 loss:{e4_loss.detach().cpu().numpy()}, e5 loss:{e5_loss.detach().cpu().numpy()}, e6 loss:{e6_loss.detach().cpu().numpy()} ')
            print(f'mu_parameter:{output1.detach().cpu().numpy()}, lambda_parameter:{output2.detach().cpu().numpy()}, M:{output3.detach().cpu().numpy()}, phi:{output4.detach().cpu().numpy()}, kappa:{output5.detach().cpu().numpy()}, alpha:{output6.detach().cpu().numpy()} ')
            print(f'w1:{w1.detach().cpu().numpy()}, w2:{w2.detach().cpu().numpy()}, w3:{w3.detach().cpu().numpy()}, w4:{w4.detach().cpu().numpy()}, w5:{w5.detach().cpu().numpy()}, w6:{w6.detach().cpu().numpy()}')



            e1_loss_list.append(e1_loss.detach().cpu().numpy())
            e2_loss_list.append(e2_loss.detach().cpu().numpy())
            e3_loss_list.append(e3_loss.detach().cpu().numpy())
            e4_loss_list.append(e4_loss.detach().cpu().numpy())
            e5_loss_list.append(e5_loss.detach().cpu().numpy())
            e6_loss_list.append(e6_loss.detach().cpu().numpy())


            w1_list.append(w1.detach().cpu().numpy())
            w2_list.append(w2.detach().cpu().numpy())
            w3_list.append(w3.detach().cpu().numpy())
            w4_list.append(w4.detach().cpu().numpy())
            w5_list.append(w5.detach().cpu().numpy())
            w6_list.append(w6.detach().cpu().numpy())




            mu_parameter_list.append(output1.detach().cpu().numpy())
            lambda_parameter_list.append(output2.detach().cpu().numpy())
            M_list.append(output3.detach().cpu().numpy())
            phi_list.append(output4.detach().cpu().numpy())
            kappa_list.append(output5.detach().cpu().numpy())
            alpha_list.append(output6.detach().cpu().numpy())


            loss_list.append(loss.detach().cpu().numpy())




        plt.figure('Loss')
        plt.subplot(1,2,1)
        plt.title('Weighted Loss')
        plt.plot(np.log10(loss_list))
        plt.subplot(1,2,2)
        plt.title('Sub Losses')
        plt.plot(e1_loss_list,'r-.', label='e1 Loss')
        plt.plot(e2_loss_list,'b--',label='e2 Loss')
        plt.plot(e3_loss_list,'g--',label='e3 Loss')
        plt.plot(e4_loss_list,'c--',label='e4 Loss')
        plt.plot(e5_loss_list,'m--',label='e5 Loss')
        plt.plot(e6_loss_list,'y--',label='e6 Loss')


        plt.legend()
        plt.savefig('./logs/loss.png')
        plt.show()


        loss_list_prediction = pd.DataFrame(loss_list)
        loss_list_prediction.to_csv("./logs/loss_list_prediction.csv")

        e1_loss_list_prediction = pd.DataFrame(e1_loss_list)
        e1_loss_list_prediction.to_csv("./logs/e1_loss_list_prediction.csv")

        e2_loss_list_prediction = pd.DataFrame(e2_loss_list)
        e2_loss_list_prediction.to_csv("./logs/e2_loss_list_prediction.csv")

        e3_loss_list_prediction = pd.DataFrame(e3_loss_list)
        e3_loss_list_prediction.to_csv("./logs/e3_loss_list_prediction.csv")

        e4_loss_list_prediction = pd.DataFrame(e4_loss_list)
        e4_loss_list_prediction.to_csv("./logs/e4_loss_list_prediction.csv")

        e5_loss_list_prediction = pd.DataFrame(e5_loss_list)
        e5_loss_list_prediction.to_csv("./logs/e5_loss_list_prediction.csv")

        e6_loss_list_prediction = pd.DataFrame(e6_loss_list)
        e6_loss_list_prediction.to_csv("./logs/e6_loss_list_prediction.csv")


        plt.figure('Total Loss')
        plt.title('Total Loss')
        plt.plot(np.log10(loss_list))
        plt.savefig('./logs/total_loss.png')
        plt.show()

        plt.figure('e1 Loss')
        plt.title('e1 Loss')
        plt.plot(np.log10(e1_loss_list))
        plt.savefig('./logs/e1_loss_list.png')
        plt.show()

        plt.figure('e2 Loss')
        plt.title('e2 Loss')
        plt.plot(np.log10(e2_loss_list))
        plt.savefig('./logs/e2_loss_list.png')
        plt.show()

        plt.figure('e3 Loss')
        plt.title('e3 Loss')
        plt.plot(np.log10(e3_loss_list))
        plt.savefig('./logs/e3_loss_list.png')
        plt.show()

        plt.figure('e4 Loss')
        plt.title('e4 Loss')
        plt.plot(np.log10(e4_loss_list))
        plt.savefig('./logs/e4_loss_list.png')
        plt.show()

        plt.figure('e5 Loss')
        plt.title('e5 Loss')
        plt.plot(np.log10(e5_loss_list))
        plt.savefig('./logs/e5_loss_list.png')
        plt.show()

        plt.figure('e6 Loss')
        plt.title('e6 Loss')
        plt.plot(np.log10(e6_loss_list))
        plt.savefig('./logs/e6_loss_list.png')
        plt.show()

        plt.figure('regularizer loss')
        plt.title('regularizer loss')
        plt.plot(np.log10(regularizer_loss_list))
        plt.savefig('./logs/regularizer_loss_list.png')
        plt.show()

        plt.figure('mu_parameter_list')
        plt.title('mu_parameter_list')
        plt.plot(mu_parameter_list)
        plt.savefig('./logs/mu_parameter_list.png')
        plt.show()

        plt.figure('lambda_parameter_list')
        plt.title('lambda_parameter_list')
        plt.plot(lambda_parameter_list)
        plt.savefig('./logs/lambda_parameter_list.png')
        plt.show()

        plt.figure('M_list')
        plt.title('M_list')
        plt.plot(M_list)
        plt.savefig('./logs/M_list.png')
        plt.show()

        plt.figure('phi_list')
        plt.title('phi_list')
        plt.plot(phi_list)
        plt.savefig('./logs/phi_list.png')
        plt.show()

        plt.figure('kappa_list')
        plt.title('kappa_list')
        plt.plot(kappa_list)
        plt.savefig('./logs/kappa_list.png')
        plt.show()

        plt.figure('alpha_list')
        plt.title('alpha_list')
        plt.plot(alpha_list)
        plt.savefig('./logs/alpha_list.png')
        plt.show()

        plt.figure('w1')
        plt.title('w1')
        plt.plot(w1_list)
        plt.savefig('./logs/w1_list.png')
        plt.show()

        plt.figure('w2')
        plt.title('w2')
        plt.plot(w2_list)
        plt.savefig('./logs/w2_list.png')
        plt.show()

        plt.figure('w3')
        plt.title('w3')
        plt.plot(w2_list)
        plt.savefig('./logs/w3_list.png')
        plt.show()

        plt.figure('w4')
        plt.title('w4')
        plt.plot(w4_list)
        plt.savefig('./logs/w4_list.png')
        plt.show()


        plt.figure('w5')
        plt.title('w5')
        plt.plot(w5_list)
        plt.savefig('./logs/w5_list.png')
        plt.show()

        plt.figure('w6')
        plt.title('w6')
        plt.plot(w6_list)
        plt.savefig('./logs/w6_list.png')
        plt.show()

        ### colorful version weights plots

        plt.figure('adapt_weights')
        plt.title('adapt_weights')
        plt.plot(w1_list,'b', label='w1')
        plt.plot(w2_list,'g',label='w2')
        plt.plot(w3_list,'r',label='w3')
        plt.plot(w4_list,'c',label='w4')
        plt.plot(w5_list,'k',label='w5')
        plt.plot(w6_list,'y',label='w6')

        plt.legend(loc='center right')
        plt.savefig('./logs/adapt_weights.png')
        plt.show()


        plt.figure('adapt_weights_without_legend')
        plt.title('adapt_weights_without_legend')
        plt.plot(w1_list,'b', label='w1')
        plt.plot(w2_list,'g',label='w2')
        plt.plot(w3_list,'r',label='w3')
        plt.plot(w4_list,'c',label='w4')
        plt.plot(w5_list,'k',label='w5')
        plt.plot(w6_list,'y',label='w6')

        plt.savefig('./logs/adapt_weights_without_legend.png')
        plt.show()


        plt.figure('log10_adapt_weights')
        plt.title('log10_adapt_weights')
        plt.plot(np.log10(w1_list),'b', label='w1')
        plt.plot(np.log10(w2_list),'g',label='w2')
        plt.plot(np.log10(w3_list),'r',label='w3')
        plt.plot(np.log10(w4_list),'c',label='w4')
        plt.plot(np.log10(w5_list),'k',label='w5')
        plt.plot(np.log10(w6_list),'y',label='w6')

        plt.legend(loc='center right')
        plt.savefig('./logs/log10_adapt_weights.png')
        plt.show()



        plt.figure('log10_adapt_weights_without_legend')
        plt.title('log10_adapt_weights_without_legend')
        plt.plot(np.log10(w1_list),'b', label='w1')
        plt.plot(np.log10(w2_list),'g',label='w2')
        plt.plot(np.log10(w3_list),'r',label='w3')
        plt.plot(np.log10(w4_list),'c',label='w4')
        plt.plot(np.log10(w5_list),'k',label='w5')
        plt.plot(np.log10(w6_list),'y',label='w6')

        plt.savefig('./logs/log10_adapt_weights_without_legend.png')
        plt.show()




if __name__ == '__main__':
    main()
