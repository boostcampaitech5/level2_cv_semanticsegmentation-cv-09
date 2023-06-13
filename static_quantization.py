import torch
import os
import argparse
import models.my_model as mm

def get_args():
    parser = argparse.ArgumentParser()
    # Data and model checkpoints directories
    parser.add_argument('--model_path', type=str, default="my_model_path", help='model.pth or model.pt')
    parser.add_argument('--model_dir', type=str, default="my_model_dir", help='../dir/dirr/dirrr/model_repository')
    
    args = parser.parse_args()
    return args

def print_model_size(model_path):
    print("%.2f MB" %(os.path.getsize(model_path)/1e6))

if __name__=="__main__":
    
    args = get_args()
    
    # Prepare input data
    input_data = torch.full((1, 3, 512, 512), 0.5).cuda()
    
    model_path = os.path.join(args.model_dir, args.model_path)
    save_path = os.path.join("./quantized_models", os.path.splitext(args.model_path)[0]+"_quantized.pt")
    model = mm.NestedUNet(deep_supervision=True).cuda()
    model.load_state_dict(torch.load(model_path).state_dict())
    # model = torch.load(model_path).cuda()
    # Make predictions
    with torch.no_grad():
        output = model(input_data)
    print("-----before quntization-----")
    print_model_size(model_path) 
    print("max : ", torch.max(output))
    print("min : ", torch.min(output))
    print("mean : ", torch.mean(output))

    backend = "qnnpack"
    model = model.cpu()
    model.qconfig = torch.quantization.get_default_qconfig(backend)
    torch.backends.quantized.engine = backend
    model_static_quantized = torch.quantization.prepare(model, inplace=False)
    model_static_quantized = torch.quantization.convert(model_static_quantized, inplace=False).cuda()
    # model_traced = torch.jit.trace(model_static_quantized, torch.quantization.QuantStub(input_data))
    # print(model_traced)
    # exit()
    model_script = torch.jit.script(model_static_quantized)
    torch.jit.save(model_script, save_path)
    model_script = torch.jit.load(save_path)
    print(torch.backends.quantized.supported_engines)
    # q_output = model_script(input_data)
    # for name, child in model_static_quantized.named_children():
    #     print(name, child)
    # exit()
    print("-----after quantization-----")
    print_model_size(save_path) 
    # print("max : ", torch.max(q_output))
    # print("min : ", torch.min(q_output))

