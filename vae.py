import math
import itertools
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim import SGD, Adam
from torch.utils.data import TensorDataset, DataLoader

class MultiOutputLinear(nn.Module):
    '''
    A module that maps a single input to multiple outputs through linear layers
    This is helpful when the network returns multple parameters, such as the mean and covariance of a Gaussian
    '''
    def __init__(self, d_in, d_out):
        super(MultiOutputLinear, self).__init__()
        for i, d in enumerate(d_out):
            self.add_module('{}'.format(i), nn.Linear(d_in, d))

    def forward(self, x):
        return [m(x) for m in self.children()]

class GaussianVAE(nn.Module):
    '''
    A module that 
    '''
    def __init__(self, encoder, decoder, L=1):
        super(GaussianVAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.L = L

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)

    def forward(self, x):
        # encode
        z_mean, z_logvar = self.encode(x)
        
        # reparameterize
        eps = Variable(torch.randn(self.L, *z_mean.size()), requires_grad=False)
        z_l = z_mean + (.5*z_logvar).exp()*eps

        # decode
        x_mean, x_logvar = self.decode(z_l)

        return z_mean, z_logvar, x_mean, x_logvar

    def elbo(self, x):
        '''
        Encodes then decodes the output
        ''' 
        return GaussianVAE_ELBO(x, *self(x))

    @property
    def input_dimension(self):
        return next(encoder.parameters()).size()[1]

    @property
    def latent_dimension(self):
        return next(decoder.parameters()).size()[1]

def GaussianVAE_ELBO(x, z_mean, z_logvar, x_mean, x_logvar):
    Dx = x_mean.size()[-1]

    # divergence between unit Gaussian and diagonal Gaussian
    divergence = - .5*torch.sum(1 + z_logvar - z_mean.pow(2) - z_logvar.exp(),1)
    divergence = torch.mean(divergence,0) # sum over all of the data in the minibatch

    # likelihood approximated by L samples of z
    log_likelihood = -torch.sum((x - x_mean)*((x - x_mean)/x_logvar.exp()),2) - Dx/2.0*(math.log(2*math.pi) + torch.sum(x_logvar,2))
    log_likelihood = torch.mean(log_likelihood,0) # average over all of the samples to approximate expectation
    log_likelihood = torch.mean(log_likelihood,0) # sum over all of the data

    return -divergence + log_likelihood

def gen_data(Dx, Dz, data_size):
    '''
    Generates random data according to a nonlinear generative model
    Z->X
    '''
    # global parameter
    W = torch.randn(Dz, Dx)
    
    # training data
    Z_train = torch.randn(N, Dz).type(torch.FloatTensor)
    eps = 0.1*torch.randn(N, Dx).type(torch.FloatTensor)
    X_train = torch.sin(Z_train).mm(W) + eps
    
    # test data
    Z_test = torch.randn(N, Dz).type(torch.FloatTensor)
    eps = 0.1*torch.randn(N, Dx).type(torch.FloatTensor)
    X_test = torch.sin(Z_test).mm(W) + eps

    
    return X_train, Z_train, X_test, Z_test

if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt

    # generate data
    N = 10000 # data size
    M = 100 # minibatch size
    Dx = 10
    Dz = 3
    x_train, z_train, x_test, z_test = gen_data(Dx, Dz, data_size=N)
    train_dataset = TensorDataset(x_train,z_train)
    test_dataset = TensorDataset(x_test,z_test)
    train_dataloader = DataLoader(train_dataset, batch_size=M)
    test_dataloader = DataLoader(test_dataset, batch_size=M)


    # setup the autoencoder
    encoder = nn.Sequential(
          nn.Linear(Dx, 100),
          nn.ReLU(),
          MultiOutputLinear(100, [Dz, Dz]),
        )

    decoder = nn.Sequential(
          nn.Linear(Dz, 100),
          nn.ReLU(),
          MultiOutputLinear(100, [Dx, Dx]),
        )

    autoencoder = GaussianVAE(encoder, decoder, L=10)

    # setup the optimizer
    learning_rate = 3e-4
    optimizer = Adam(autoencoder.parameters(), lr=learning_rate)

    # optimize
    num_epochs = 200
    elbo_train = np.zeros(num_epochs)
    elbo_test = np.zeros(num_epochs)
    for epoch in range(num_epochs):

        # compute test ELBO
        for batch_i, batch in enumerate(test_dataloader):
            data = Variable(batch[0], requires_grad=False)
            elbo_test[epoch] += autoencoder.elbo(data).data[0]
        elbo_test[epoch] /= len(test_dataloader)

        # compute training ELBO
        for batch_i, batch in enumerate(train_dataloader):
            data = Variable(batch[0], requires_grad=False)

            autoencoder.zero_grad()
            loss = -autoencoder.elbo(data)
            loss.backward()
            elbo_train[epoch] += -loss.data[0]
            optimizer.step()
        elbo_train[epoch] /= len(train_dataloader)
        print('Epoch [{}/{}]\
               \n\tTrain ELBO: {}\n\tTest ELBO:  {}'.format(\
                epoch+1, num_epochs, \
                elbo_train[epoch], elbo_test[epoch]))

    print(elbo_train)
    plt.plot(elbo_train, label='training Lower Bound')
    plt.plot(elbo_test, label='test Lower Bound')
    plt.legend()
    plt.show()