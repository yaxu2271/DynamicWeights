import torch
import torch.nn as nn
from pdb import set_trace as st
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def poro_loss (ip, ip_x, ip_xx, ip_y, ip_yy, iux, iux_x, iux_xx, iux_xy, iux_yy, iuy, iuy_xx, iuy_y, iuy_yx, iuy_yy, rp, rp_x, rp_xx, rp_y, rp_yy, rux, rux_x, rux_xx, rux_xy, rux_yy, ruy, ruy_xx, ruy_y, ruy_yx, ruy_yy, rF1x, iF1x, rF1y, iF1y, rF2, iF2, output1, output2, output3, output4, output5, output6):
    
    
    ## hyperparameter eta setup to adjust the loss balancing rate for each components
    eta = 1
    ## material parameter setup
    rho = 2.27
    rho_f = 1
    rho_a = 0.117
    w = 391
    mu_parameter = output1.to(device)
    lambda_parameter = output2.to(device)

    M = output3.to(device)

    phi = output4.to(device)
    rgamma = rho_a/(phi**2) + rho_f/phi

    kappa = output5.to(device)
    igamma = 1/(w*kappa)


    alpha = output6.to(device)


    ## set up artificial parameters to simplify calculation process
    ra = alpha - rho_f*rgamma/(rgamma**2+igamma**2)
    ia = rho_f*igamma/(rgamma**2+igamma**2)
    rb = rho - rho_f**2*rgamma/(rgamma**2+igamma**2)
    ib = rho_f**2*igamma/(rgamma**2+igamma**2)
    rc = rgamma/(rgamma**2+igamma**2)
    ic = -igamma/(rgamma**2+igamma**2)
    
    
    
    mu_parameter_order = torch.round(torch.log10(mu_parameter))
    lame_parameter_order = torch.round(torch.log10(mu_parameter + lambda_parameter))
    M_order = torch.round(torch.log10(M))
    ra_order = torch.round(torch.log10(torch.abs(ra)))
    ia_order = torch.round(torch.log10(torch.abs(ia)))
    rb_order = torch.round(torch.log10(torch.abs(rb)))
    ib_order = torch.round(torch.log10(torch.abs(ib)))
    rc_order = torch.round(torch.log10(torch.abs(rc)))
    ic_order = torch.round(torch.log10(torch.abs(ic)))
    
    



    # original loss function for each PDE
    e1 = ((mu_parameter*(rux_xx+rux_yy)+(lambda_parameter + mu_parameter)*(rux_xx+ruy_yx)-ra*rp_x+ia*ip_x+w**2*rb*rux-w**2*ib*iux)-rF1x)
    e2 = ((mu_parameter*(iux_xx+iux_yy)+(lambda_parameter + mu_parameter)*(iux_xx+iuy_yx)-ra*ip_x-ia*rp_x+w**2*rb*iux+w**2*ib*rux)-iF1x)
    e3 = ((mu_parameter*(ruy_xx+ruy_yy)+(lambda_parameter + mu_parameter)*(rux_xy+ruy_yy)-ra*rp_y+ia*ip_y+w**2*rb*ruy-w**2*ib*iuy)-rF1y)
    e4 = ((mu_parameter*(iuy_xx+iuy_yy)+(lambda_parameter + mu_parameter)*(iux_xy+iuy_yy)-ra*ip_y-ia*rp_y+w**2*rb*iuy+w**2*ib*ruy)-iF1y)
    e5 = ((1/w**2)*rc*(rp_xx+rp_yy)-(1/w**2)*ic*(ip_xx+ip_yy)+rp/M+ra*(rux_x+ruy_y)-ia*(iux_x+iuy_y)-rF2)
    e6 = ((1/w**2)*rc*(ip_xx+ip_yy)+(1/w**2)*ic*(rp_xx+rp_yy)+ip/M+ra*(iux_x+iuy_y)+ia*(rux_x+ruy_y)-iF2)


    ## define magnitude indicator for each subloss
    
    
    # e1
    e11 = torch.round(torch.log10(torch.mean(torch.abs(rux_xx + rux_yy))))+mu_parameter_order
    e12 = torch.round(torch.log10(torch.mean(torch.abs(rux_xx + ruy_yx))))+lame_parameter_order
    e13 = torch.round(torch.log10(torch.mean(torch.abs(rp_x))))+ra_order
    e14 = torch.round(torch.log10(torch.mean(torch.abs(ip_x))))+ia_order
    e15 = torch.round(torch.log10(torch.mean(torch.abs(rux))*w**2))+rb_order
    e16 = torch.round(torch.log10(torch.mean(torch.abs(iux))*w**2))+ib_order
    e17 = torch.round(torch.log10(torch.mean(torch.abs(rF1x))))
    ll1 = 10**((e11+e12+e13+e14+e15+e16+e17)/7)
    

    # e2
    e21 = torch.round(torch.log10(torch.mean(torch.abs(iux_xx+iux_yy))))+mu_parameter_order
    e22 = torch.round(torch.log10(torch.mean(torch.abs(iux_xx+iuy_yx))))+lame_parameter_order
    e23 = torch.round(torch.log10(torch.mean(torch.abs(ip_x))))+ra_order
    e24 = torch.round(torch.log10(torch.mean(torch.abs(rp_x))))+ia_order
    e25 = torch.round(torch.log10(torch.mean(torch.abs(iux))*w**2))+rb_order
    e26 = torch.round(torch.log10(torch.mean(torch.abs(rux))*w**2))+ib_order
    e27 = torch.round(torch.log10(torch.mean(torch.abs(iF1x))))
    ll2 = 10**((e21+e22+e23+e24+e25+e26+e27)/7)
    

    # e3
    e31 = torch.round(torch.log10(torch.mean(torch.abs(ruy_xx+ruy_yy))))+mu_parameter_order
    e32 = torch.round(torch.log10(torch.mean(torch.abs(rux_xy+ruy_yy))))+lame_parameter_order
    e33 = torch.round(torch.log10(torch.mean(torch.abs(rp_y))))+ra_order
    e34 = torch.round(torch.log10(torch.mean(torch.abs(ip_y))))+ia_order
    e35 = torch.round(torch.log10(torch.mean(torch.abs(ruy))*w**2))+rb_order
    e36 = torch.round(torch.log10(torch.mean(torch.abs(iuy))*w**2))+ib_order
    e37 = torch.round(torch.log10(torch.mean(torch.abs(rF1y))))
    ll3 = 10**((e31+e32+e33+e34+e35+e36)/6)
    # note e37 = 0
    
    # e4
    e41 = torch.round(torch.log10(torch.mean(torch.abs(iuy_xx+iuy_yy))))+mu_parameter_order
    e42 = torch.round(torch.log10(torch.mean(torch.abs(iux_xy+iuy_yy))))+lame_parameter_order
    e43 = torch.round(torch.log10(torch.mean(torch.abs(ip_y))))+ra_order
    e44 = torch.round(torch.log10(torch.mean(torch.abs(rp_y))))+ia_order
    e45 = torch.round(torch.log10(torch.mean(torch.abs(iuy))*w**2))+rb_order
    e46 = torch.round(torch.log10(torch.mean(torch.abs(ruy))*w**2))+ib_order
    e47 = torch.round(torch.log10(torch.mean(torch.abs(iF1y))))
    # note e47 = 0


    ll4 = 10**((e41+e42+e43+e44+e45+e46)/6)
    

    # e5
    e51 = torch.round(torch.log10(torch.mean(torch.abs(rp_xx+rp_yy))/(w**2)))+rc_order
    e52 = torch.round(torch.log10(torch.mean(torch.abs(ip_xx+ip_yy))/(w**2)))+ic_order
    e53 = torch.round(torch.log10(torch.mean(torch.abs(rp))))-M_order
    e54 = torch.round(torch.log10(torch.mean(torch.abs(rux_x+ruy_y))))+ra_order
    e55 = torch.round(torch.log10(torch.mean(torch.abs(iux_x+iuy_y))))+ia_order
    e56 = torch.round(torch.log10(torch.mean(torch.abs(rF2))))

    ll5 = 10**((e51+e52+e53+e54+e55+e56)/6)

    # e6
    e61 = torch.round(torch.log10(torch.mean(torch.abs(ip_xx+ip_yy))/(w**2)))+rc_order
    e62 = torch.round(torch.log10(torch.mean(torch.abs(rp_xx+rp_yy))/(w**2)))+ic_order
    e63 = torch.round(torch.log10(torch.mean(torch.abs(ip))))-M_order
    e64 = torch.round(torch.log10(torch.mean(torch.abs(iux_x+iuy_y))))+ra_order
    e65 = torch.round(torch.log10(torch.mean(torch.abs(rux_x+ruy_y))))+ia_order
    e66 = torch.round(torch.log10(torch.mean(torch.abs(iF2))))

    ll6 = 10**((e61+e62+e63+e64+e65+e66)/6)
    
    ll1 = ll1**eta 
    ll2 = ll2**eta 
    ll3 = ll3**eta 
    ll4 = ll4**eta 
    ll5 = ll5**eta 
    ll6 = ll6**eta 

    
    w1 = (1/ll1)**2
    w2 = (1/ll2)**2
    w3 = (1/ll3)**2
    w4 = (1/ll4)**2
    w5 = (1/ll5)**2
    w6 = (1/ll6)**2
    
    ww = w1 + w2 + w3 + w4 + w5 + w6
    
    w1 = w1/ww
    w2 = w2/ww
    w3 = w3/ww
    w4 = w4/ww
    w5 = w5/ww
    w6 = w6/ww


    
    
    e1_loss = w1*torch.mean(e1**2)
    e2_loss = w2*torch.mean(e2**2)
    e3_loss = w3*torch.mean(e3**2)
    e4_loss = w4*torch.mean(e4**2)
    e5_loss = w5*torch.mean(e5**2)
    e6_loss = w6*torch.mean(e6**2)
    

        
    loss = e1_loss + e2_loss + e3_loss + e4_loss + e5_loss + e6_loss



    return loss, e1_loss, e2_loss, e3_loss, e4_loss, e5_loss, e6_loss, w1, w2, w3, w4, w5, w6
