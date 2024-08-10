import numpy as np
import torch
if __name__ == '__main__':
    mean_file = 'H:/code/python/IRFishDetection2.0.0/dataset2.1/anji_process/pca/t3/svdFeature_mean.npy'
    sigma_file = 'H:/code/python/IRFishDetection2.0.0/dataset2.1/anji_process/pca/t3/svdFeature_Sigma.npy'
    U_file = 'H:/code/python/IRFishDetection2.0.0/dataset2.1/anji_process/pca/t3/svdFeature_U.npy'
    pca_param_save_path = 'H:/code/python/IRFishDetection2.0.0/results/svd_feature/used_for_train/svd_features_2.5_1_1'
    svd_mean = torch.from_numpy(np.load(mean_file)).unsqueeze(0)
    svd_sigma = torch.from_numpy(np.load(sigma_file))
    svd_U = torch.from_numpy(np.load(U_file))
    svd_U_t = torch.from_numpy(np.load(U_file)).transpose(1,0)
    pca_param = {'mean_feature': svd_mean,
                 'svd_feature': svd_U_t,
                 'singular_values': svd_sigma}
    torch.save(pca_param, pca_param_save_path)
    pass