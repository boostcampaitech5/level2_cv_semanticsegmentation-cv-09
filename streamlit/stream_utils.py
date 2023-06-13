import sys
sys.path.append('../')
import numpy as np
import torch
import torch.nn.functional as F
import cv2

CLASSES = [
        'finger-1', 'finger-2', 'finger-3', 'finger-4', 'finger-5',
        'finger-6', 'finger-7', 'finger-8', 'finger-9', 'finger-10',
        'finger-11', 'finger-12', 'finger-13', 'finger-14', 'finger-15',
        'finger-16', 'finger-17', 'finger-18', 'finger-19', 'Trapezium',
        'Trapezoid', 'Capitate', 'Hamate', 'Scaphoid', 'Lunate',
        'Triquetrum', 'Pisiform', 'Radius', 'Ulna',
        ]
PALETE = [
            (220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228),
            (0, 60, 100), (0, 80, 100), (0, 0, 70), (0, 0, 192), (250, 170, 30),
            (100, 170, 30), (220, 220, 0), (175, 116, 175), (250, 0, 30), (165, 42, 42),
            (255, 77, 255), (0, 226, 252), (182, 182, 255), (0, 82, 0), (120, 166, 157),
            (110, 76, 0), (174, 57, 255), (199, 100, 0), (72, 0, 118), (255, 179, 240),
            (0, 125, 92), (209, 0, 151), (188, 208, 182), (0, 220, 176),
        ]

CLASS2IND = {v: i for i, v in enumerate(CLASSES)}
IND2CLASS = {v: k for k, v in CLASS2IND.items()}


def label2rgb(label):
    image_size = label.shape[1:] + (3, )
    image = np.zeros(image_size, dtype=np.uint8)
    
    for i, class_label in enumerate(label):
        image[class_label == 1] = PALETE[i]
        
    return image


def encode_mask_to_rle(mask):
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def decode_rle_to_mask(rle, height, width):
    s = rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(height * width, dtype=np.uint8)
    
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    
    return img.reshape(height, width)


def out2img(output):
    
    output = F.interpolate(output, size=(2048, 2048), mode="bilinear")
    output = torch.sigmoid(output)
    output = (output > 0.5).detach().cpu().numpy()

    outs = []
    for out in output:
        for segm in out:
            rle = encode_mask_to_rle(segm)
            outs.append(rle)
    
    preds = []
    for rle in outs:
        pred = decode_rle_to_mask(rle, height=2048, width=2048)
        preds.append(pred)
    
    preds = np.stack(preds,0)
    
    out_img = label2rgb(preds)
    
    return out_img


def do_gradcam(model,img):
    
    # find last output, last layer
    global last_layer
    global last_grad
    last_layer = {}
    last_grad = 0
    handles = []
    def hook(name):
        def hook_fn(module,input,output):
            # last_layer['module'] = module
            global last_layer
            last_layer['name'] = name
            last_layer['output'] = output.detach()
        
        return hook_fn

    def grad_hook(module, grad_input, grad_output):
        global last_grad
        last_grad = grad_output[0].detach()
        
    for name, module in model.named_modules():
        
        if isinstance(module, torch.nn.Conv2d):
            forward_hook = module.register_forward_hook(hook(name))
            backward_hook = module.register_full_backward_hook(grad_hook)
            handles.extend([forward_hook, backward_hook])
    
    # foward, backward
    with torch.autograd.set_grad_enabled(True):
        output = model(img)
        model.zero_grad()
        output.backward(torch.ones_like(output))
    
    for handle in handles:
        handle.remove()
    
    # calculrate gradcam
    result_gradcam = []
    weights = last_grad.sum(dim=-1,keepdim=True).sum(dim=-2,keepdim=True)
    for id in range(29):
        weighted_activations = (weights *last_layer['output'][0][id]).sum(dim=1)
        cam = F.relu((weighted_activations)).squeeze(0)
        normal_cam = (cam - cam.min())/cam.max()
        result_cam = cv2.resize(normal_cam.cpu().data.numpy(), (img.size(-2),img.size(-1)))
        result_gradcam.append(result_cam)
    
    if img.shape[-2:] != output.shape[-2:]:
        output = F.interpolate(output, size= img.shape[-2:],mode='bilinear')
    
    return result_gradcam, output 


def label2mask(annotations, label):
    for ann in annotations:
        c = ann["label"]
        class_ind = CLASS2IND[c]
        points = np.array(ann["points"])
        
        # polygon to mask
        class_label = np.zeros(label.shape[:2], dtype=np.uint8)
        cv2.fillPoly(class_label, [points], 1)
        label[..., class_ind] = class_label
    return label


def compare_loss(rgb_img, mask,output):
    if mask.shape[-2:] != output.shape[-2:]:
        output = F.interpolate(output, size= mask.shape[-2:],mode='bilinear')
        
    losses = F.binary_cross_entropy_with_logits(output,mask,reduction='none').squeeze(0)
    losses = (losses-losses.min())/losses.max()
    loss_maps = []
    gt_maps = []
    
    for i in range(29):
        # calculrate loss map
        loss = losses[i].detach().cpu().numpy()
        heatmap = cv2.applyColorMap(np.uint8(255 *loss), cv2.COLORMAP_JET)
        lossmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        mix_loss = np.uint8(0.5*rgb_img*255 + 0.5*lossmap)
        loss_maps.append(mix_loss)

        # make gt
        gt = mask[0][i].detach().cpu().numpy()
        gt_img = label2rgb(np.expand_dims(gt,axis=0))
        mix_gt = np.uint8(0.5*rgb_img*255 + 0.5*gt_img)
        gt_maps.append(mix_gt)
    
    return gt_maps, loss_maps

def gt_predict_diff(mask,output):
    if mask.shape[-2:] != output.shape[-2:]:
        output = F.interpolate(output, size= mask.shape[-2:],mode='bilinear')
    output = torch.sigmoid(output)
    output = (output > 0.5).squeeze(0).detach().cpu().numpy().astype(float)
    
    mask = mask.squeeze(0).detach().cpu().numpy()
    diff_img = np.sum(np.abs(mask-output),axis=0)
    diff_img = diff_img/ np.max(diff_img)
    gts = label2rgb(mask)
    
    return gts, np.uint8(diff_img*255)
    