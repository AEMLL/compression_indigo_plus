import yaml
import os

if __name__ == '__main__':
    scale = 3.5
    in_dir = 'configs/sample/large_zeta30'
    out_dir = f'configs/sample/large_zeta{round(scale*10)}'
    qf = range(0,20)
    for factor in qf:
        filename = f'indigo_syn_jpeg_qf{factor}.yaml'
        
        path = os.path.join(in_dir, filename)
        with open(path, 'r') as file:
            config = yaml.safe_load(file)

        print(config)

        config['scale'] = scale
        config['INN_dir'] = f"weights/INN/large/INN_JPEG_QF{factor}.pth"

        path = os.path.join(out_dir, filename)
        with open(path, 'w') as file:
            yaml.dump(config, file)