import torch
import numpy as np
import os
from networks.generator import Generator
from load_volume_data_RCA import Dataset
from samples_parameters import SAMPLES_PARA

# Paths
ab_path = os.getcwd() + '/DeepCA/'
weights_path = ab_path + 'outputs_results/checkpoints/Epoch_20.tar'
output_dir = ab_path + 'output_test/'
os.makedirs(output_dir, exist_ok=True)

BATCH_SIZE = 3

def load_model(device):
	model = Generator(in_channels=1, num_filters=64, class_num=1).to(device)
	checkpoint = torch.load(weights_path, map_location=device)
	model.load_state_dict(checkpoint['network'])
	model.eval()
	return model

def run_inference(model, dataloader, device):
	results = []
	threshold = 0.5  # You can change this value as needed
	with torch.no_grad():
		for idx, data in enumerate(dataloader):
			inputs, labels = data[0].float().to(device), data[1].float().to(device)
			outputs = model(inputs)
			# Apply threshold to outputs to get binary mask
			outputs_bin = (outputs > threshold).float()
			for i in range(outputs_bin.shape[0]):
				out_np = outputs_bin[i].cpu().numpy().astype(np.uint8)  # Save as 0/1 uint8
				label_np = labels[i].cpu().numpy().astype(np.uint8)
				input_np = inputs[i].cpu().numpy().astype(np.uint8)
				np.save(os.path.join(output_dir, f'output_{idx}_{i}.npy'), out_np)
				np.save(os.path.join(output_dir, f'label_{idx}_{i}.npy'), label_np)
				np.save(os.path.join(output_dir, f'input_{idx}_{i}.npy'), input_np)
				results.append((out_np, label_np, input_np))
	return results

def main():
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model = load_model(device)
	test_set = Dataset(SAMPLES_PARA['test_index'])
	testloader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=1, drop_last=False)
	run_inference(model, testloader, device)
	print(f"Inference complete. Results saved to {output_dir}")

if __name__ == "__main__":
	main()
