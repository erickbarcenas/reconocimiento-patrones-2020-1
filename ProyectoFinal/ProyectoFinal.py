#!/usr/bin/env python
# coding: utf-8

# # Proyecto Final. Generación de sintética de tomografías.
# 
# Reconocomiento de Patrones
# 
# Andrés González Flores
# 

# In[1]:


from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
import glob
import imageio
import codecs
import hashlib 
import time

# Set random seed for reproducibility
manualSeed = 999
#manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)


# ## Inputs
# 
# Let’s define some inputs for the run:
# 
# - **dataroot** - the path to the root of the dataset folder. We will talk more about the dataset in the next section
# - **workers** - the number of worker threads for loading the data with the DataLoader
# - **batch_size** - the batch size used in training. The DCGAN paper uses a batch size of 128
# - **image_size** - the spatial size of the images used for training. This implementation defaults to 64x64. If another size is desired, the structures of D and G must be changed. See here for more details
# - **nc** - number of color channels in the input images. For color images this is 3
# - **nz** - length of latent vector
# - **ngf** - relates to the depth of feature maps carried through the generator
# - **ndf** - sets the depth of feature maps propagated through the discriminator
# - **num_epochs** - number of training epochs to run. Training for longer will probably lead to better results but will also take much longer
# - **lr** - learning rate for training. As described in the DCGAN paper, this number should be 0.0002
# - **beta1** - beta1 hyperparameter for Adam optimizers. As described in paper, this number should be 0.5
# - **ngpu** - number of GPUs available. If this is 0, code will run in CPU mode. If this number is greater than 0 it will run on that number of GPUs

# In[2]:


# Root directory for dataset
dataroot = "dataset"

# Number of workers for dataloader
workers = 4

# Batch size during training
batch_size = 128

# Spatial size of training images. All images will be resized to this
#   size using a transformer.
image_size = 64

# Number of channels in the training images. For color images this is 3
nc = 1

# Size of z latent vector (i.e. size of generator input)
nz = 100

# Size of feature maps in generator
ngf = 64

# Size of feature maps in discriminator
ndf = 64

# Number of training epochs
num_epochs = 1000

# Learning rate for optimizers
dlr = 0.00005 
glr = 0.0002 

# Beta1 hyperparam for Adam optimizers
beta1 = 0.5

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1

# Imágenes para animación gif
nimgif = 12


# ## Datos
# 
# En esta práctica usaré imágenes de tomografías.

# In[3]:


# We can use an image folder dataset the way we have it setup.
# Create the dataset
dataset = dset.ImageFolder(root=dataroot,
                           transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.Grayscale(num_output_channels=1),
                               transforms.ToTensor(),
                               transforms.Normalize([0.5], [0.5]),
                           ]))

# Número de imágenes en el dataset
n_imgs = len(dataset.imgs)

# Create the dataloader
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=workers)

# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

# Para generar imágenes de muestra cuadradas
if batch_size<64:
    vector_cuad = [x**2 for x in range(1, 9)]
    indice = np.floor(np.sqrt(32)).astype(np.uint8)-1
    n_img_grid = vector_cuad[indice]
else:
    n_img_grid = 64

# Plot some training images
real_batch = next(iter(dataloader))
plt.figure(figsize=(8,8))
plt.axis("off")
plt.title("Training Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:n_img_grid], padding=2, normalize=True).cpu(),(1,2,0)))
plt.show()


# ## Implementación
# 
# ### Inicialización de pesos
# Del artículo de DCGAN, los autores indican que los pesos deben ser inicializados aleatoreamente de una distribución Normal con media = 0 y desviación estándar = 0.02. La función weights_init function toma un modelo inicializado como entrada y reinicializa todas sus capas para que cumplan con este criterio. La función es aplicada a los modelos inmediatamente después de su inicialización.

# In[4]:


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


# ### Generator
# The generator, G, is designed to map the latent space vector (z) to data-space. Since our data are images, converting z to data-space means ultimately creating a RGB image with the same size as the training images (i.e. 3x64x64). In practice, this is accomplished through a series of strided two dimensional convolutional transpose layers, each paired with a 2d batch norm layer and a relu activation. The output of the generator is fed through a tanh function to return it to the input data range of \[−1,1\]. It is worth noting the existence of the batch norm functions after the conv-transpose layers, as this is a critical contribution of the DCGAN paper. These layers help with the flow of gradients during training. An image of the generator from the DCGAN paper is shown below.
# 
# ![Generador](dcgan_generator.png)
# 
# Notice, the how the inputs we set in the input section (nz, ngf, and nc) influence the generator architecture in code. nz is the length of the z input vector, ngf relates to the size of the feature maps that are propagated through the generator, and nc is the number of channels in the output image (set to 3 for RGB images). Below is the code for the generator.

# In[5]:


# Generator Code
class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(in_channels = nz, 
                               out_channels = ngf * 8, 
                               kernel_size = 4, 
                               stride = 1, 
                               padding = 0, 
                               bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)


# Now, we can instantiate the generator and apply the weights_init function. Check out the printed model to see how the generator object is structured.

# In[6]:


# Create the generator
netG = Generator(ngpu).to(device)

# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    netG = nn.DataParallel(netG, list(range(ngpu)))

# Apply the weights_init function to randomly initialize all weights
#  to mean=0, stdev=0.2.
netG.apply(weights_init)

# Print the model
print(netG)


# ### Discriminator
# 
# As mentioned, the discriminator, D, is a binary classification network that takes an image as input and outputs a scalar probability that the input image is real (as opposed to fake). Here, D takes a 3x64x64 input image, processes it through a series of Conv2d, BatchNorm2d, and LeakyReLU layers, and outputs the final probability through a Sigmoid activation function. This architecture can be extended with more layers if necessary for the problem, but there is significance to the use of the strided convolution, BatchNorm, and LeakyReLUs. The DCGAN paper mentions it is a good practice to use strided convolution rather than pooling to downsample because it lets the network learn its own pooling function. Also batch norm and leaky relu functions promote healthy gradient flow which is critical for the learning process of both G and D.
# 
# Discriminator Code

# In[7]:


class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)


# Now, as with the generator, we can create the discriminator, apply the weights_init function, and print the model’s structure.

# In[8]:


# Create the Discriminator
netD = Discriminator(ngpu).to(device)

# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    netD = nn.DataParallel(netD, list(range(ngpu)))

# Apply the weights_init function to randomly initialize all weights
#  to mean=0, stdev=0.2.
netD.apply(weights_init)

# Print the model
print(netD)


# Loss Functions and Optimizers
# With D and G setup, we can specify how they learn through the loss functions and optimizers. We will use the Binary Cross Entropy loss (BCELoss) function which is defined in PyTorch as:
# 
# ℓ(x,y)=L={l<sub>1</sub>,…,l<sub>N</sub>}<sup>T</sup>,l<sub>n</sub>=−\[y<sub>n</sub>⋅logx<sub>n</sub>+(1−y<sub>n</sub>)⋅log(1−x<sub>n</sub>)\]
# 
# Notice how this function provides the calculation of both log components in the objective function (i.e. log(D(x)) and log(1−D(G(z)))). We can specify what part of the BCE equation to use with the y input. This is accomplished in the training loop which is coming up soon, but it is important to understand how we can choose which component we wish to calculate just by changing y (i.e. GT labels).
# 
# Next, we define our real label as 1 and the fake label as 0. These labels will be used when calculating the losses of D and G, and this is also the convention used in the original GAN paper. Finally, we set up two separate optimizers, one for D and one for G. As specified in the DCGAN paper, both are Adam optimizers with learning rate 0.0002 and Beta1 = 0.5. For keeping track of the generator’s learning progression, we will generate a fixed batch of latent vectors that are drawn from a Gaussian distribution (i.e. fixed_noise) . In the training loop, we will periodically input this fixed_noise into G, and over the iterations we will see images form out of the noise.

# In[9]:


# Initialize BCELoss function
criterion = nn.BCELoss()

# Create batch of latent vectors that we will use to visualize
#  the progression of the generator
fixed_noise = torch.randn(64, nz, 1, 1, device=device)

# Establish convention for real and fake labels during training
real_label = 1
fake_label = 0

# Setup Adam optimizers for both G and D
optimizerD = optim.Adam(netD.parameters(), lr=dlr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=glr, betas=(beta1, 0.999))


# ### Training
# Finally, now that we have all of the parts of the GAN framework defined, we can train it. Be mindful that training GANs is somewhat of an art form, as incorrect hyperparameter settings lead to mode collapse with little explanation of what went wrong. Here, we will closely follow Algorithm 1 from Goodfellow’s paper, while abiding by some of the best practices shown in ganhacks. Namely, we will “construct different mini-batches for real and fake” images, and also adjust G’s objective function to maximize logD(G(z)). Training is split up into two main parts. Part 1 updates the Discriminator and Part 2 updates the Generator.
# 
# #### Part 1 - Train the Discriminator
# 
# Recall, the goal of training the discriminator is to maximize the probability of correctly classifying a given input as real or fake. In terms of Goodfellow, we wish to “update the discriminator by ascending its stochastic gradient”. Practically, we want to maximize log(D(x))+log(1−D(G(z))). Due to the separate mini-batch suggestion from ganhacks, we will calculate this in two steps. First, we will construct a batch of real samples from the training set, forward pass through D, calculate the loss (log(D(x))), then calculate the gradients in a backward pass. Secondly, we will construct a batch of fake samples with the current generator, forward pass this batch through D, calculate the loss (log(1−D(G(z)))), and accumulate the gradients with a backward pass. Now, with the gradients accumulated from both the all-real and all-fake batches, we call a step of the Discriminator’s optimizer.
# 
# #### Part 2 - Train the Generator
# 
# As stated in the original paper, we want to train the Generator by minimizing log(1−D(G(z))) in an effort to generate better fakes. As mentioned, this was shown by Goodfellow to not provide sufficient gradients, especially early in the learning process. As a fix, we instead wish to maximize log(D(G(z))). In the code we accomplish this by: classifying the Generator output from Part 1 with the Discriminator, computing G’s loss using real labels as GT, computing G’s gradients in a backward pass, and finally updating G’s parameters with an optimizer step. It may seem counter-intuitive to use the real labels as GT labels for the loss function, but this allows us to use the log(x) part of the BCELoss (rather than the log(1−x) part) which is exactly what we want.
# 
# Finally, we will do some statistic reporting and at the end of each epoch we will push our fixed_noise batch through the generator to visually track the progress of G’s training. The training statistics reported are:
# 
# - Loss_D - discriminator loss calculated as the sum of losses for the all real and all fake batches (log(D(x))+log(D(G(z)))).
# - Loss_G - generator loss calculated as log(D(G(z)))
# - D(x) - the average output (across the batch) of the discriminator for the all real batch. This should start close to 1 then theoretically converge to 0.5 when G gets better. Think about why this is.
# - D(G(z)) - average discriminator outputs for the all fake batch. The first number is before D is updated and the second number is after D is updated. These numbers should start near 0 and converge to 0.5 as G gets better. Think about why this is.
# 
# Note: This step might take a while, depending on how many epochs you run and if you removed some data from the dataset.

# In[10]:


# Training Loop

# Lists to keep track of progress
img_list = []
G_losses = []
D_losses = []
iters = 0

# Guardar imágenes cada i iteraciones
modulo = np.ceil(num_epochs*np.round(n_imgs/batch_size)/(nimgif-1))

print("Starting Training Loop...")
# For each epoch
start = time.time()
for epoch in range(num_epochs):
    # For each batch in the dataloader
    for i, data in enumerate(dataloader, 0):

        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        ## Train with all-real batch
        netD.zero_grad()
        # Format batch
        real_cpu = data[0].to(device)
        b_size = real_cpu.size(0)
        label = torch.full((b_size,), real_label, device=device)
        # Forward pass real batch through D
        output = netD(real_cpu).view(-1)
        # Calculate loss on all-real batch
        errD_real = criterion(output, label)
        # Calculate gradients for D in backward pass
        errD_real.backward()
        D_x = output.mean().item()

        ## Train with all-fake batch
        # Generate batch of latent vectors
        noise = torch.randn(b_size, nz, 1, 1, device=device)
        # Generate fake image batch with G
        fake = netG(noise)
        label.fill_(fake_label)
        # Classify all fake batch with D
        output = netD(fake.detach()).view(-1)
        # Calculate D's loss on the all-fake batch
        errD_fake = criterion(output, label)
        # Calculate the gradients for this batch
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        # Add the gradients from the all-real and all-fake batches
        errD = errD_real + errD_fake
        # Update D
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = netD(fake).view(-1)
        # Calculate G's loss based on this output
        errG = criterion(output, label)
        # Calculate gradients for G
        errG.backward()
        D_G_z2 = output.mean().item()
        # Update G
        optimizerG.step()

        # Output training stats
        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, num_epochs, i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        # Save Losses for plotting later
        G_losses.append(errG.item())
        D_losses.append(errD.item())

        # Check how the generator is doing by saving G's output on fixed_noise
        if (iters % modulo == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()
            img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

        iters += 1
        
end = time.time()
t_entrenamiento = end-start


# **Tiempo de entrenamiento**

# In[11]:


(t_ent, seg) = divmod(np.round(t_entrenamiento), 60) 
(t_ent, minut) = divmod(t_ent, 60)
(t_ent, horas) = divmod(t_ent, 60) 
(t_ent, dias) = divmod(t_ent, 24) 
print('Tiempo de entrenamiento: ', end='')
if dias > 0:
    print('%d día(s), ' % dias, end='')
if horas > 0:
    print('%d hora(s), ' % horas, end='')
if minut > 0:
    print('%d minuto(s), ' % minut, end='')
print('%d segundo(s)' % seg)


# ## Results
# 
# Finally, lets check out how we did. Here, we will look at three different results. First, we will see how D and G’s losses changed during training. Second, we will visualize G’s output on the fixed_noise batch for every epoch. And third, we will look at a batch of real data next to a batch of fake data from G.
# 
# **Loss versus training iteration** 
# 
# Below is a plot of D & G’s losses versus training iterations.

# In[12]:


plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses,label="G")
plt.plot(D_losses,label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()


# **Visualization of G’s progression**
# 
# Remember how we saved the generator’s output on the fixed_noise batch after every epoch of training. Now, we can visualize the training progression of G with an animation. Press the play button to start the animation.

# In[13]:


#%%capture
fig = plt.figure(figsize=(8,8))
plt.axis("off")
ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True, cmap='gray')] for i in img_list]
ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)

HTML(ani.to_jshtml())


# **Guardo la imagen en formato gif**

# In[14]:


gif_file = 'resultado.gif'

imagenes = []
imagenes_gif = []
for img in img_list:
    img = np.dstack((img.numpy()*256).astype(np.uint8))
    imagenes.append(img)
    for j in range(7):
        imagenes_gif.append(img)
                    
imagenes  = np.array(imagenes)
#imagenes_gif  = np.array(imagenes)

imageio.mimsave(gif_file, imagenes_gif)


# **Guardando resultado en formato avi**

# In[15]:


import cv2

video_file = 'resultado.avi'

height, width, layers = imagenes[0].shape
size = (width,height)

out = cv2.VideoWriter(video_file,cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
 
for i in range(len(imagenes)):
    for j in range(15):
        out.write(imagenes[i])
out.release()


# **Real Images vs. Fake Images**
# 
# Finally, lets take a look at some real images and fake images side by side.

# In[16]:


# Grab a batch of real images from the dataloader
real_batch = next(iter(dataloader))

# Plot the real images
fig = plt.figure(figsize=(15,15))
plt.subplot(1,2,1)
plt.axis("off")
plt.title("Imágenes Reales")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(),(1,2,0)))

# Plot the fake images from the last epoch
plt.subplot(1,2,2)
plt.axis("off")
plt.title("Imágenes Falsas")
plt.imshow(np.transpose(img_list[-1],(1,2,0)))
plt.show()

# Guardar la comparación
comp_file = 'real vs fake.png'
fig.savefig(comp_file)


# **Guardando hyperparámetros**

# In[17]:


# Función hash para nombres de folders únicos
def md5(fname):
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

md_file = 'hyperparametros.md'

with codecs.open(md_file, 'w', 'utf-8') as archivo:
    archivo.write('# Parámetros usados para obtener este resultado\n\n')
    archivo.write('- manualSeed = %d\n' % manualSeed)
    archivo.write('- batch_size = %d\n' % batch_size)
    archivo.write('- image_size = %d\n' % image_size)
    archivo.write('- nz = %d\n' % nz)
    archivo.write('- ngf = %d\n' % ngf)
    archivo.write('- ndf = %d\n' % ndf)
    archivo.write('- num_epochs = %d\n' % num_epochs)
    #archivo.write('- lr = %d\n' % lr)
    archivo.write('- dlr = %f\n' % dlr)
    archivo.write('- glr = %f\n' % glr)
    archivo.write('- beta1 = %f\n' % beta1)

# Creo folder con nombre único basado en los hiperparámetros
folder = md5(md_file)
try:
    os.mkdir('./resultados/'+folder)
    print('Folder creado')
except FileExistsError:
    print('El folder ya existe')


# **Guardo diccionarios de estado**

# In[29]:


f_models = 'checkpoint.pth' 
torch.save({
            'D_state_dict': netD.state_dict(),
            'G_state_dict': netG.state_dict(),
            #'optimizerD_state_dict': optimizerD.state_dict(),
            #'optimizerG_state_dict': optimizerG.state_dict(),
            }, f_models)


# **Muevo todos los archivos al folder creado**

# In[30]:


f_list = [md_file,gif_file,video_file,comp_file, f_models]
for fname in f_list:    
    try:    
        os.rename(fname, './resultados/'+folder+'/'+fname)
    except FileExistsError as fexerr:
        print("FileExistsError: {0}".format(fexerr))
    except PermissionError as perr:
        print("PermissionError: {0}".format(perr))
    except FileNotFoundError as fnferr:
        print("FileNotFoundError: {0}".format(fnferr))
                             


# In[ ]:




