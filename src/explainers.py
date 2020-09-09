import cv2
import numpy as np
import math

import torch
from torch import nn
from torch.autograd import Variable
from tqdm import tqdm
try:
    from utils import SaveOutput
except:
    from src.utils import SaveOutput

class GradCam():

    def __init__(self, model, module, device):
        self.device = device
        self.module = module.to(device)
        self.model = model.to(device)

    def __call__(self, *args, **kwargs):
        return self.compute_gradcam(*args, **kwargs)

    def get_hook(self):
        forward = SaveOutput()
        back = SaveOutput()
        handle1 = self.module.register_forward_hook(forward)
        handle2 = self.module.register_backward_hook(back)
        return forward, back

    def get_values(self, inputs, target):
        inputs = inputs.to(self.device)
        forward, back = self.get_hook()
        x = inputs.clone()
        if len(x.shape) == 3:
            x = x.reshape(1, *list(inputs.shape))

        #print(x.shape)
        out = self.model(x)
        out.view(-1)[target].backward()
        activation, gradient = forward.outputs[0], back.outputs[0]
        return activation, gradient[0]

    def compute_gradcam(self, inputs, target,
                        out_type = 'id',
                        interpolation_mode=cv2.INTER_LANCZOS4,
                        additional_out = False,
                        verbose = 0):
        activation, gradient = self.get_values(inputs, target)
        alpha = torch.sum(gradient, dim=(0, 2, 3))
        C = alpha.view(-1, 1, 1) * activation[0]
        out = torch.sum(C, dim=(0))

        if verbose:
            print('Input:\n', inputs.shape)
            print('Gradient:\n', gradient.shape)
            print('Alpha:\n', alpha.shape)
            print('Activation:\n', activation.shape)
            print('alpha*A:\n', C.shape)
            print('Out:\n', out.shape)

        out = self.output_processing(out, out_type)
        resized_out = self.resize(inputs, out, interpolation_mode=interpolation_mode)
        if additional_out:
            return resized_out.view(inputs.shape[-2], inputs.shape[-1]), out
        return resized_out.view(inputs.shape[-2], inputs.shape[-1])

    def resize(self, input, output, interpolation_mode):
        dim = tuple(input.shape[-2:])
        resized_img = cv2.resize(output.numpy(), dim, interpolation = interpolation_mode)
        return torch.tensor(resized_img)

    def output_processing(self, out, out_type):
        if out_type == 'id':
            return out.detach().cpu()
        elif out_type == 'relu':
            return nn.ReLU()(out).detach().cpu()
        elif out_type == 'abs':
            return torch.abs(out).detach().cpu()
        else:
            raise NotImplementedError


def smoothing(method, X, iter, var, **kwargs):
    results = torch.Tensor([])
    for i in range(iter):
        x = X + torch.randn(*list(X.shape))*var
        output = method(**kwargs)
        results = torch.cat((results, torch.tensor(np.mean(cycle, axis=0)).view(1, 28, 28)))


"""
Exception to catch Vanishing Gradient Problem caused by ReLU.
Present in standard VGG, problem fixed in VGG with Batch Normalization.
"""
class GradientIsZeroError(Exception):
    pass

"""
The loss to minimize:
fn_type:        Decide what loss function to use. If 0 minimize by MSE given a target activation, else minimze -x (maximise x). Default 1
max_activation: The target activation that we want to reach. Deafult is 5.
"""
class FilterLoss(nn.Module):

    def __init__(self, fn_type=1,  max_activation=5):
        super().__init__()
        self.fn_type = fn_type
        self.max_activation = max_activation

    """
    The forward function is called in __call__(self, x) on the nn.Module. Calling FilterLoss()(x) will call forward(x).
    Return the desired loss.
    """
    def forward(self, x):
        if self.fn_type:
            return -x
        return 0.5*(self.max_activation-x)**2


"""
The class that hook the output of a module of a model. A module is simply a layer.
"""
class SaveFeatures():
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
    def hook_fn(self, module, input, output):
        self.features = output
    def close(self):
        self.hook.remove()


"""
Detector Class that perform Gradient Ascent w.r.t. an input in order to maximise the output of a filter (neuron) of a convolutional layer
"""
class Detector():

    """
    Init mode
    """
    def __init__(self, model, loss_type = 1, lr=0.1, device=torch.device('cuda')):
        """
        model: A Pytorch Neural Network based on nn.Module. In our case VGG16
        loss_type: The loss function we want to use. See Class FilterLoss for further detail. Deafualt is 1.
        lr: learning rate we want to use. Deafult is 0.1. TODO: probably better to move it in detect fucntion
        """
        self.model = model
        self.loss_type = loss_type
        self.lr = lr
        self.device = device
    """
    Call the gradient ascent procedure on a given neuron of a given layer.
    """
    def detect(self, dim, module, neuron,
               max_activation = 5, max_epoch=100,
               project = False,
               verbose=0, debug=False, warm_start=False, tqdm_disable=False):

        """
        NAME              TYPE           DESCRIPTION

        INPUT VARS:
        dim:              tuple.         A two dimensional tuple indicating the height and width of the input image.
        sequential:       int.           The VGG target Sequential module. VGG has 3 sequential modules:
                                         0: Convolutional Module (only tested on it)
                                         1: Adaptive Pooling Module
                                         2: Linear Module
        layer:            int.           The target layer of the target sequential module.
        neuron:           int.           The target neuron of the target layer.
        max_activation:   float.         The activation we want to reach. Deafault is 5.
        max_epoch:        int.           The maximum number of iteration. Default is 100
        verbose:          int/bool.      If 1/True print the loss and the output on each iteration. Deafult is 0/False
        debug:            int/bool.      If 1/True print additional info. Default is 0/False
        warm_start:       False/Tensor.  An already generated input to use as starting point instead of a random one. Used in upscaling, if nothing is passed (False) a random generated input is used.

        OUTPUT VARS:
        loss:             float.         The loss reached. Hopefully close to 0 if the max activation is not too high and/or the number of epochs is high enough
        out:              float.         The output of the target neuron given the last input
        self.x:           Tensor.        The final input that should maximise the activation
        """

        ### 1. Define the loss Function and the information of the target neuron (neuron, layer, sequential)
        self.loss_fn = FilterLoss(max_activation = max_activation, fn_type=self.loss_type)
        self.neuron, self.module = neuron, module
        self.verbose = verbose

        ### 2. Init a Variable with Gradient from either a random input or a given input (see warm_start)
        if type(warm_start) == bool:
            # Here we generate a random input with pixel scaled from 0.40 to 0.60
            self.x = Variable(torch.randn([1, 3, dim[0], dim[1]]).to(self.device), requires_grad = True)
        else:
            self.x = Variable(torch.tensor(warm_start).float().to(self.device), requires_grad = True)

        ### 3. Put model in evaluation mode in order to be sure to not train the weights and to put dropout in non mask mode
        self.model.eval()
        if verbose or debug:
            tqdm_disable = verbose or debug


        ### 4. Call the SGD for the desired number of epoch or until the target activation is
        for ep in tqdm(range(max_epoch), disable=tqdm_disable):
            ### 4.1 Run SGD
            loss, out = self.SGD(debug, project)
            ### 4.2 Check for Gradient Vanish Problem
            if math.isnan(loss):
                raise GradientIsZeroError('Output is 0, gradient is 0')
            ### 4.3 Print info
            if verbose:
                print('Epoch {}: {}\t{}'.format(ep+1, loss, out))
            ### 4.4 Check if the activation is reached
            if out > (max_activation-0.05):

                break


        ### 5. Return
        hook = SaveFeatures(self.module)
        out = self.model(self.x)                                            # Feed the input to the model
        loss = self.loss_fn(hook.features.mean(dim=(2, 3))[0][self.neuron])   # Compute the loss using the output of the neuron saved in "hook.features"
        #print(loss, hook.features.mean(dim=(2, 3))[0][self.neuron])
        return loss, out, self.x


    """
    Run Gradient Descent w.r.t the input
    """
    def SGD(self, debug, project):
        ### 1. Get the target layer and tell the SaveFeatures Class to save the output of that layer
        hook = SaveFeatures(self.module)                                                                     # When an input will be fed to the model, the SaveFeatures class will hook the output of out layer and store them in SaveFeatures.features.

        ### 2.Compute Loss and gradient
        out = self.model(self.x) # Feed the input to the model

        loss = self.loss_fn(hook.features.mean(dim=(2, 3))[0][self.neuron])                             # Compute the loss using the output of the neuron saved in "hook.features"
        loss.backward(retain_graph=True)                                                                # Call the backward function on the loss. It will compute the gradient of all the variables with gradient attached. i.e. our input

        ########
        # DEBUG
        ########
        if debug:
            print('Output: {}'.format(hook.features.mean(dim=(2, 3))[0][self.neuron]))
            print('Loss: {}'.format(loss))
            if torch.norm(self.x.grad) > 30:
                print('Grad shape: {}'.format(self.x.grad.shape))
                print('Norm: {}'.format(torch.norm(self.x.grad)))
                print('Grad: {}'.format(self.x.grad[0, 0, 0, :10]))
                print('Descent Direction:{}'.format(self.lr*(self.x.grad/torch.norm(self.x.grad))[0, 0, 0, :10]))
            else:
                print('Norm: {}'.format(torch.norm(self.x.grad)))
                print('Descent Direction:{}'.format((self.lr*self.x.grad)[0, 0, 0, :10]))

        ### 3.  Update fase: the gradient tends to be to small so if this happens we normalized it.
        ### 3.1 Update input
        if torch.norm(self.x.grad) > 50:
            self.x.data -= self.lr*self.x.grad/torch.norm(self.x.grad)
        #elif torch.norm(self.x.grad) < 1:
        #    self.x.data -= self.lr*self.x.grad/torch.norm(self.x.grad)
        else:
            self.x.data -= self.lr*self.x.grad
        ### 3.2 Use Projecting Gradient Descent. Easy since out hypercube is defined over our basis
        if project:
            self.x.data[self.x < project[0]] = project[0]
            self.x.data[self.x > project[1]] = project[1]


        ### 4. Remove the output
        hook.close()

        return loss, hook.features.mean(dim=(2, 3))[0][self.neuron]


"""
Visualizer Class.
Given a model and a dictionary of parameters containing as key 'dim', 'neuron' and 'layer' it will allow you to:
    1. Get the activations of the neurons of layer given an input x.                  get_hook(x)[1]
    2. Get the outputs (NxN matrix, img) of the neurons of a layer given an input x.  get_hook(x)[0]
    3. Plot the activations, the input and the outpus of a layer given input x.       visualize(x)
"""
class Visualizer():

    """
    NAME        TYPE        DESCRIPTION
    model       nn.Module   The target network. VGG16
    params      dict        A dictionary containing the following pair of (key, value):
                            1. key: 'dim'       value: tuple of (height, width) of the input img
                            2. key: 'neuron'    value: number of the neuron in the target layer
                            3. key: 'layer'     value: number of the layer
    """
    def __init__(self, model, params):
        self.model = model
        self.params  = params

    """
    Feed to the model the input x and register the outputs of all the layers under the dictionary self.features (conv layers) and self.dense (linear layer). Return the output (useless??)
    """
    def hook(self, x):
        ### 1. Init variables
        self.x = x
        features = {}
        dense = {}

        ### 2. Define private functions
        def hook_fn1(m, i, o):
            features[m] = o
        def hook_fn2(m, i, o):
            dense[m] = o

        ### 3. Save the hooked outputs in the dictionaries with keys the name of the layer
        for name, layer in self.model.features._modules.items():
            layer.register_forward_hook(hook_fn1)
        for name, layer in self.model.classifier._modules.items():
            layer.register_forward_hook(hook_fn2)

        ### 4. Feed the network
        out = self.model(x)

        ### 5. Save the dictionaries as Class Attributes
        self.features = features
        self.dense = dense

        return out

    """
    Call the hook on a given input and return the activations and the outputs of that layer
    """
    def get_hook(self, x):
        ### 1. Call the hook
        target_out = self.hook(x)

        ### 2. Get the outpus and compute the activations (the mean of the ouputs pixels)
        keys = [k for k in self.features.keys()]
        self.out = self.features[keys[self.params['layer']]][0]
        self.acts = np.round(torch.mean(self.out, dim=(1, 2)).detach().cpu().numpy(), 3)

        ### 3. Return outpus and activations
        return self.out, self.acts

    """
    The model accepts torch.tensor of shape (1, 3, n, n) while to display the image we need a numpy.array of shape (n, n, 3).
    So here we convert the input from torch.tensor to numpy.arrray and we swap the axis.
    """
    def swap_axis(self):
        ### 1. Convert it input to numpy
        output = self.x.detach().cpu().view(3, self.params['dim'][0], self.params['dim'][0]).numpy()

        ### 2. Swap axis (maybe there is a more intelligent way??)
        mat = np.zeros([self.params['dim'][0], self.params['dim'][0], 3])
        for n, channel in enumerate(output):
            mat[:, :, n] = channel

        ### 3. Return the ready to be displayed numpy.array
        return mat

    """
    Here we visualize a series of information:

    1. The 1st plot is a plt.bar visualizing the activation of the target neuron and of its neighboors.
    2. The 2nd plot are two images:
            a. The img of output of the conv target neuron
            b. The img of the input which maximise the activation of the target neuron
    3. The outputs of the neurons of the target layer.
    """
    def visualize(self, x):
        ### 0. Get output and activations of the target_layer
        _, _ = self.get_hook(x)


        start = max(0, self.params['neuron']-32)
        end = min(self.params['neuron']+32, len(self.acts))

        plt.rcParams["figure.figsize"] = (5,3)
        #### 1. Activations
        fig = plt.figure(figsize=(18, 5))
        fig.add_subplot(111)
        plt.bar([str(i) for i in range(start, end)], self.acts[start:end])
        plt.xticks(rotation=90)
        plt.grid()

        ### 2. Target feature_map/outpus and input
        fig = plt.figure(figsize=(18, 9))
        plt.suptitle('Mean activation: {:0.4f}'.format(round(self.acts[self.params['neuron']], 3)))
        # 2.a
        fig.add_subplot(121)
        plt.imshow(self.out[self.params['neuron']].detach().cpu().numpy())
        plt.title('Neuron Map')
        # 2.b
        fig.add_subplot(122)
        mat = self.swap_axis()
        plt.imshow(mat)
        plt.title('Input')

        ### 3. All features_map/outputs
        fig = plt.figure(figsize=(25, 25))
        for i in range(6):
            for j in range(6):
                fig.add_subplot(6, 6, i*6+j+1)
                plt.imshow(self.out[start+i*6+j].detach().cpu().numpy())
                plt.title('Neuron: {}\nMean activation: {:0.4f}'.format(start+i*6+j, round(self.acts[start+i*6+j], 3)))
        plt.tight_layout()
        plt.show()

"""
Save fig in current path calling it 'fmap_l"layer_number"_n"neuron_number".png' if no other img of same title are present.
If same name images are present call it 'fmap_l"layer_number"_n"neuron_number"v"version_number".png'
"""
def save_fig(img):
    total_number = len(str(len(acts)))
    neuron_number = str(params['neuron'])
    while len(neuron_number) < total_number:
        neuron_number = '0{}'.format(neuron_number)
    name='fmap_l{}_n{}.png'.format(params['layer'], neuron_number)

    n=2
    print()
    while name in os.listdir():
        name='fmap_l{}_n{}v{}.png'.format(params['layer'], neuron_number, n)
        n += 1
    plt.savefig(name)
