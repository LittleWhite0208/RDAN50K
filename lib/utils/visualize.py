import numpy as np
import os
import torch, torchvision

to_np = lambda x: x.data.cpu().numpy()

def write_observations(config, epoch, x, x_r):
	# store samples ..
	if epoch==0:
		torchvision.utils.save_image(x.data, '%s/origion.png' % (config.img_path),
									 normalize=True)

	# samples = v.Decoder(z_fixed)
	elif (epoch+1) % 10 ==0:
		torchvision.utils.save_image(x_r.data, '%s/%03d.png' % (config.img_path, epoch / config.snapshot), normalize=True)
	else:
		pass
	# .. checkpoint ..
	# torch.save(v.state_dict(), os.path.join(config.ckpt_path, 'v.pth'))

	# .. and collect loss
	# logger["loss"] = np.append(logger["loss"], to_np(loss))