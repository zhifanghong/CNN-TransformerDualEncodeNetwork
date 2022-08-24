import os, time
import matplotlib.pyplot as plt
import itertools
import pickle
import imageio
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# G(z)
class generator(nn.Module):
    # initializers
    def __init__(self, d=128):
        super(generator, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(100, d*8, 4, 1, 0) #4
        self.deconv1_bn = nn.BatchNorm2d(d*8)
        self.deconv2 = nn.ConvTranspose2d(d*8, d*4, 4, 2, 1)  #8
        self.deconv2_bn = nn.BatchNorm2d(d*4)
        self.deconv3 = nn.ConvTranspose2d(d*4, d*2, 4, 2, 1)  #16
        self.deconv3_bn = nn.BatchNorm2d(d*2)
        self.deconv4 = nn.ConvTranspose2d(d*2, d, 4, 2, 1)  #32
        self.deconv4_bn = nn.BatchNorm2d(d)
        self.deconv5 = nn.ConvTranspose2d(d, d, 4, 2, 1)   #64
        self.deconv5_bn = nn.BatchNorm2d(d)
        self.deconv6 = nn.ConvTranspose2d(d, d, 4, 2, 1)   #128
        self.deconv6_bn = nn.BatchNorm2d(d)
        self.deconv7 = nn.ConvTranspose2d(d, d, 4, 2, 1)   #256
        self.deconv7_bn = nn.BatchNorm2d(d)
        self.deconv8 = nn.ConvTranspose2d(d, 2, 4, 2, 1)   #512

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input):
        # x = F.relu(self.deconv1(input))
        x = F.relu(self.deconv1_bn(self.deconv1(input)))
        x = F.relu(self.deconv2_bn(self.deconv2(x)))
        x = F.relu(self.deconv3_bn(self.deconv3(x)))
        x = F.relu(self.deconv4_bn(self.deconv4(x)))
        x = F.relu(self.deconv5_bn(self.deconv5(x)))
        x = F.relu(self.deconv6_bn(self.deconv6(x)))
        x = F.relu(self.deconv7_bn(self.deconv7(x)))
        x = torch.tanh(self.deconv8(x))

        return x

class discriminator(nn.Module):
    # initializers
    def __init__(self, d=128):
        super(discriminator, self).__init__()
        self.conv1 = nn.Conv2d(2, d, 4, 2, 1)
        self.conv2 = nn.Conv2d(d, d, 4, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(d)
        self.conv3 = nn.Conv2d(d, d, 4, 2, 1)
        self.conv3_bn = nn.BatchNorm2d(d)
        self.conv4 = nn.Conv2d(d, d, 4, 2, 1)
        self.conv4_bn = nn.BatchNorm2d(d)
        self.conv5 = nn.Conv2d(d, d*2, 4, 2, 1)
        self.conv5_bn = nn.BatchNorm2d(d*2)
        self.conv6 = nn.Conv2d(d*2, d*4, 4, 2, 1)
        self.conv6_bn = nn.BatchNorm2d(d*4)
        self.conv7 = nn.Conv2d(d*4, d*8, 4, 2, 1)
        self.conv7_bn = nn.BatchNorm2d(d*8)
        self.conv8 = nn.Conv2d(d*8, 1, 4, 1, 0)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input):
        x = F.leaky_relu(self.conv1(input), 0.2)
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.conv4_bn(self.conv4(x)), 0.2)
        x = F.leaky_relu(self.conv5_bn(self.conv5(x)), 0.2)
        x = F.leaky_relu(self.conv6_bn(self.conv6(x)), 0.2)
        x = F.leaky_relu(self.conv7_bn(self.conv7(x)), 0.2)
        x = torch.sigmoid(self.conv8(x))

        return x

def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()

fixed_z_ = torch.randn((5 * 5, 100)).view(-1, 100, 1, 1)    # fixed noise
fixed_z_ = Variable(fixed_z_.cuda(), volatile=True)
def show_result(num_epoch, show = False, save = False, path = 'result.png', isFix=False):
    z_ = torch.randn((5*5, 100)).view(-1, 100, 1, 1)
    z_ = Variable(z_.cuda(), volatile=True)

    G.eval()
    if isFix:
        test_images = G(fixed_z_)
    else:
        test_images = G(z_)
    G.train()
    #print(test_images.shape)
    size_figure_grid = 5
    fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(5, 5))
    for i, j in itertools.product(range(size_figure_grid), range(size_figure_grid)):
        ax[i, j].get_xaxis().set_visible(False)
        ax[i, j].get_yaxis().set_visible(False)

    for k in range(5*5):
        i = k // 5
        j = k % 5
        ax[i, j].cla()
        ax[i, j].imshow(test_images[k, 0].cpu().data.numpy(), cmap='gray')
        #ax[2*i+1, j].imshow(test_images[k, 1].cpu().data.numpy(), cmap='gray')

    label = 'Epoch {0}'.format(num_epoch)
    fig.text(0.5, 0.04, label, ha='center')
    plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()

def show_train_hist(hist, show = False, save = False, path = 'Train_hist.png'):
    x = range(len(hist['D_losses']))

    y1 = hist['D_losses']
    y2 = hist['G_losses']

    plt.plot(x, y1, label='D_loss')
    plt.plot(x, y2, label='G_loss')

    plt.xlabel('Iter')
    plt.ylabel('Loss')

    plt.legend(loc=4)
    plt.grid(True)
    plt.tight_layout()

    if save:
        plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()

# training parameters
batch_size = 24
lr = 0.002
train_epoch = 1000


# data_loader
img_size = 64
transform = transforms.Compose([
    #transforms.Resize(img_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])
# train_loader = torch.utils.data.DataLoader(
#     datasets.MNIST('data', train=True, download=True, transform=transform),
#     batch_size=batch_size, shuffle=True)
from datasets.dataset_synapse import Synapse_dataset,RandomGenerator
from torch.utils.data import DataLoader
import random
db_train = Synapse_dataset(base_dir="./data/Synapse/train_npz", list_dir='./lists/lists_Synapse', split="train",
                transform=transforms.Compose([RandomGenerator(output_size=[512, 512])]))
def worker_init_fn(worker_id):
    random.seed(1234 + worker_id)
train_loader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True,
                         worker_init_fn=worker_init_fn)
# network
G = generator(64)
D = discriminator(64)
G.weight_init(mean=0.0, std=0.02)
D.weight_init(mean=0.0, std=0.02)
G.cuda()
D.cuda()

# Binary Cross Entropy loss
BCE_loss = nn.BCELoss()

# Adam optimizer
G_optimizer = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
D_optimizer = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

# results save folder
if not os.path.isdir('synapse_DCGAN_results'):
    os.mkdir('synapse_DCGAN_results')
if not os.path.isdir('synapse_DCGAN_results/Random_results'):
    os.mkdir('synapse_DCGAN_results/Random_results')
if not os.path.isdir('synapse_DCGAN_results/Fixed_results'):
    os.mkdir('synapse_DCGAN_results/Fixed_results')

train_hist = {}
train_hist['D_losses'] = []
train_hist['G_losses'] = []
train_hist['per_epoch_ptimes'] = []
train_hist['total_ptime'] = []
num_iter = 0

print('training start!')
start_time = time.time()
from tqdm import tqdm
for epoch in tqdm(range(train_epoch)):
    D_losses = []
    G_losses = []
    epoch_start_time = time.time()
    for _ , x_ in enumerate(train_loader):
        #print(x_["image"].shape)
        #print(x_["label"].shape)
        x_ = torch.cat([x_["image"],x_["label"].view(-1,1,512,512)],dim=1)
        #print(x_.shape)
        # train discriminator D
        D.zero_grad()

        mini_batch = x_.size()[0]

        y_real_ = torch.ones(mini_batch)
        y_fake_ = torch.zeros(mini_batch)

        x_, y_real_, y_fake_ = Variable(x_.cuda()), Variable(y_real_.cuda()), Variable(y_fake_.cuda())
        D_result = D(x_).squeeze()
        #print(D_result.shape)
        D_real_loss = BCE_loss(D_result, y_real_)

        z_ = torch.randn((mini_batch, 100)).view(-1, 100, 1, 1)
        z_ = Variable(z_.cuda())
        G_result = G(z_)

        #print(G_result.shape)

        D_result = D(G_result).squeeze()
        D_fake_loss = BCE_loss(D_result, y_fake_)
        D_fake_score = D_result.data.mean()

        D_train_loss = D_real_loss + D_fake_loss

        D_train_loss.backward()
        D_optimizer.step()

        # D_losses.append(D_train_loss.data[0])
        D_losses.append(D_train_loss.item())

        # train generator G
        G.zero_grad()

        z_ = torch.randn((mini_batch, 100)).view(-1, 100, 1, 1)
        z_ = Variable(z_.cuda())

        G_result = G(z_)
        D_result = D(G_result).squeeze()
        G_train_loss = BCE_loss(D_result, y_real_)
        G_train_loss.backward()
        G_optimizer.step()

        #G_losses.append(G_train_loss.data[0])
        G_losses.append(G_train_loss.item())

        num_iter += 1

    epoch_end_time = time.time()
    per_epoch_ptime = epoch_end_time - epoch_start_time


    print('[%d/%d] - ptime: %.2f, loss_d: %.3f, loss_g: %.3f' % ((epoch + 1), train_epoch, per_epoch_ptime, torch.mean(torch.FloatTensor(D_losses)),
                                                                 torch.mean(torch.FloatTensor(G_losses))))
    p = 'synapse_DCGAN_results/Random_results/synapse_DCGAN_' + str(epoch + 1) + '.png'
    fixed_p = 'synapse_DCGAN_results/Fixed_results/synapse_DCGAN_' + str(epoch + 1) + '.png'
    show_result((epoch+1), save=True, path=p, isFix=False)
    show_result((epoch+1), save=True, path=fixed_p, isFix=True)
    train_hist['D_losses'].append(torch.mean(torch.FloatTensor(D_losses)))
    train_hist['G_losses'].append(torch.mean(torch.FloatTensor(G_losses)))
    train_hist['per_epoch_ptimes'].append(per_epoch_ptime)

end_time = time.time()
total_ptime = end_time - start_time
train_hist['total_ptime'].append(total_ptime)

print("Avg per epoch ptime: %.2f, total %d epochs ptime: %.2f" % (torch.mean(torch.FloatTensor(train_hist['per_epoch_ptimes'])), train_epoch, total_ptime))
print("Training finish!... save training results")
torch.save(G.state_dict(), "synapse_DCGAN_results/generator_param.pkl")
torch.save(D.state_dict(), "synapse_DCGAN_results/discriminator_param.pkl")
with open('synapse_DCGAN_results/train_hist.pkl', 'wb') as f:
    pickle.dump(train_hist, f)

show_train_hist(train_hist, save=True, path='synapse_DCGAN_results/synapse_DCGAN_train_hist.png')

images = []
for e in range(train_epoch):
    img_name = 'synapse_DCGAN_results/Fixed_results/synapse_DCGAN_' + str(e + 1) + '.png'
    images.append(imageio.imread(img_name))
imageio.mimsave('synapse_DCGAN_results/generation_animation.gif', images, fps=5)